import os, random, argparse, sys, pickle, time, datasets, transformers
import torch
from transformers import AutoConfig, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from models.modelings_alignable_llama import *
from models.modelings_alignable_gpt2 import *
from models.modelings_gpt2 import *
from logic_data.constants import *
from datasets import Dataset 
from torch.utils.data import DataLoader
import gc
import wandb
from dataclasses import dataclass, field
from transformers import (
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed
)
from transformers import Trainer

# Enable DDP then FSDP
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import logging
logging.basicConfig(level = logging.INFO)

###############################
# 
# GPT-2 small related code
# 
# 
# 
###############################

class LogicSolverTrainer(object):
    def __init__(
        self, model,
        is_master,
        device,
        logger,
        lr=5e-5,
        apex_enable=False,
        n_gpu=1,
        early_stopping=5,
        do_statistic=False,
        is_wandb=False,
        model_name="",
    ):
        self.model = model
        self.is_master = is_master
        self.logger = logger
        self.is_wandb = is_wandb
        self.model_name = model_name
        
        self.device = device
        self.lr = lr
        self.n_gpu = n_gpu
    
        self.early_stopping = early_stopping
    
    def train(
        self, train_dataloader, dev_dataloader,
        optimizer, scheduler, output_dir,
        log_step, valid_steps, epochs, 
        gradient_accumulation_steps,
    ):
        self.model.train()
        train_iterator = trange(
            0, int(epochs), desc="Epoch"
        )
        total_step = 0
        total_log_step = 0
        best_eval_acc = -1
        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc=f"Epoch: {epoch}", position=0, leave=True)
            for step, inputs in enumerate(epoch_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                outputs = self.model(**inputs)
                loss = outputs.loss.mean() if self.n_gpu > 1 else outputs.loss
                
                actual_test_labels = inputs['labels'][:, -3]
                pred_test_labels = torch.argmax(outputs.logits[:, -4], dim=-1)
                correct_labels = (actual_test_labels==pred_test_labels)
                
                step_accuracy = correct_labels.sum() / correct_labels.shape[0]
                step_accuracy = step_accuracy.tolist()

                if total_step % log_step == 0 and self.is_wandb:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/step_accuracy": step_accuracy
                        },
                        step=total_log_step
                    )
                    
                    if total_step % valid_steps == 0:
                        total_count = 0
                        correct_count = 0
                        self.model.eval()
                        for step, inputs in enumerate(dev_dataloader):
                            for k, v in inputs.items():
                                if v is not None and isinstance(v, torch.Tensor):
                                    inputs[k] = v.to(self.device)
                            outputs = self.model(**inputs)

                            actual_test_labels = inputs['labels'][:, -3]
                            pred_test_labels = torch.argmax(outputs.logits[:, -4], dim=-1)
                            correct_labels = (actual_test_labels==pred_test_labels)

                            total_count += len(correct_labels)
                            correct_count += correct_labels.sum().tolist()

                        current_acc = round(correct_count/total_count, 2)
                        wandb.log(
                            {
                                "eval/accuracy": current_acc
                            },
                            step=total_log_step
                        )
                        if current_acc > best_eval_acc:
                            best_eval_acc = current_acc
                            if self.is_master:
                                if self.n_gpu > 1:
                                    self.model.module.save_pretrained(os.path.join(output_dir, 'model-best'))
                                else:
                                    self.model.save_pretrained(os.path.join(output_dir, 'model-best'))
                        self.model.train()
                        
                    
                    total_log_step += 1
                loss_str = round(loss.item(), 2)
                epoch_iterator.set_postfix({'loss': loss_str})
                
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                
                if total_step % gradient_accumulation_steps == 0:
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    
                total_step += 1
                
        logging.info("Training is finished ...") 
        if self.is_master:
            if self.n_gpu > 1:
                self.model.module.save_pretrained(os.path.join(output_dir, 'model-last'))
            else:
                self.model.save_pretrained(os.path.join(output_dir, 'model-last'))

class LogicSolverAligner(object):
    def __init__(
        self, model,
        is_master,
        device,
        logger,
        lr=5e-5,
        apex_enable=False,
        n_gpu=1,
        early_stopping=5,
        do_statistic=False,
        is_wandb=False,
        model_name="",
        intervention_config=None
    ):
        self.model = model
        self.is_master = is_master
        self.logger = logger
        self.is_wandb = is_wandb
        self.model_name = model_name
        
        self.device = device
        self.lr = lr
        self.n_gpu = n_gpu
    
        self.early_stopping = early_stopping
    
        self.intervention_config = intervention_config
        self.preload_intervention_corr = None
        # this is to make things a little faster.
        if len(list(self.intervention_config.keys())) == 1:
            self.preload_intervention_corr = self.intervention_config[
                list(self.intervention_config.keys())[0]
            ]
            self.preload_intervention_corr = torch.tensor(self.preload_intervention_corr).long()
    
    def train(
        self, train_dataloader, dev_dataloader,
        optimizer, scheduler, output_dir,
        log_step, valid_steps, epochs, 
        gradient_accumulation_steps,
    ):
        # okay, have to honest, not sure whether we do train mode align or eval align;
        # i guess it is good to try both, but ... only trying train here and move on.
        self.model.train()
        train_iterator = trange(
            0, int(epochs), desc="Epoch"
        )
        total_step = 0
        total_log_step = 0
        best_eval_acc = -1
        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc=f"Epoch: {epoch}", position=0, leave=True)
            for step, inputs in enumerate(epoch_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                        
                if self.preload_intervention_corr is not None:
                    intervention_corr = self.preload_intervention_corr.expand(
                        inputs['input_ids'].shape[0],-1
                    ).to(self.device)
                else:
                    assert False # not implemented
                
                # aligning forward!
                source_hidden_states = self.model(
                   input_ids=inputs['source_input_ids']
                ).rotated_hidden_states
                
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    source_hidden_states=source_hidden_states,
                    intervention_corr=intervention_corr,
                    labels=inputs['counterfactual_labels']
                )
                loss = outputs.loss.mean() if self.n_gpu > 1 else outputs.loss
                
                actual_test_labels = inputs['counterfactual_labels'][:, -3]
                pred_test_labels = torch.argmax(outputs.logits[:, -4], dim=-1)
                correct_labels = (actual_test_labels==pred_test_labels)
                step_accuracy = correct_labels.sum() / correct_labels.shape[0]
                step_accuracy = step_accuracy.tolist()

                if total_step % log_step == 0 and self.is_wandb:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/step_accuracy": step_accuracy
                        },
                        step=total_log_step
                    )
                    
                    if total_step % valid_steps == 0:
                        total_count = 0
                        correct_count = 0
                        self.model.eval()
                        for step, inputs in enumerate(dev_dataloader):
                            for k, v in inputs.items():
                                if v is not None and isinstance(v, torch.Tensor):
                                    inputs[k] = v.to(self.device)
                            if self.preload_intervention_corr is not None:
                                intervention_corr = self.preload_intervention_corr.expand(
                                    inputs['input_ids'].shape[0],-1
                                ).to(self.device)
                            else:
                                assert False # not implemented

                            # aligning forward!
                            source_hidden_states = self.model(
                               input_ids=inputs['source_input_ids']
                            ).rotated_hidden_states
                            outputs = self.model(
                                input_ids=inputs['input_ids'],
                                source_hidden_states=source_hidden_states,
                                intervention_corr=intervention_corr,
                                labels=inputs['counterfactual_labels']
                            )

                            actual_test_labels = inputs['counterfactual_labels'][:, -3]
                            pred_test_labels = torch.argmax(outputs.logits[:, -4], dim=-1)
                            correct_labels = (actual_test_labels==pred_test_labels)

                            total_count += len(correct_labels)
                            correct_count += correct_labels.sum().tolist()

                        current_acc = round(correct_count/total_count, 2)
                        wandb.log(
                            {
                                "eval/accuracy": current_acc
                            },
                            step=total_log_step
                        )
                        if current_acc > best_eval_acc:
                            best_eval_acc = current_acc
                            if self.is_master:
                                if self.n_gpu > 1:
                                    self.model.module.save_pretrained(os.path.join(output_dir, 'model-best'))
                                else:
                                    self.model.save_pretrained(os.path.join(output_dir, 'model-best'))
                        self.model.train()
                        
                    
                    total_log_step += 1
                loss_str = round(loss.item(), 2)
                epoch_iterator.set_postfix({'loss': loss_str})
                
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                
                if total_step % gradient_accumulation_steps == 0:
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    
                total_step += 1
                
        logging.info("Training is finished ...") 
        if self.is_master:
            if self.n_gpu > 1:
                self.model.module.save_pretrained(os.path.join(output_dir, 'model-last'))
            else:
                self.model.save_pretrained(os.path.join(output_dir, 'model-last'))
                

###############################
# 
# Alpaca related code
# 
# 
# 
###############################
"""
This code is designed for alignment search
for large models, i.e., >1B parameters.

We test it out with Alpaca 7B which is based
on LLaMA 7B model, but it should be extensible
to larger models as well if computation resource
is allowed.
"""
CACHE_DIR = "../.cache/"
    
alpaca_prompt_template = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
%s

### Input:
%s

### Response:
"""

def sample_number_by_region(
    region, lower_bound, upper_bound,
    min_lower_bound, max_upper_bound,
    excludes=[]
):
    number = None
    while number is None or number in excludes:
        if region == 1:
            number = round(random.uniform(min_lower_bound, lower_bound), 2)
        elif region == 2:
            number = round(random.uniform(lower_bound, upper_bound), 2)
        elif region == 3:
            number = round(random.uniform(upper_bound, max_upper_bound), 2)
    return number
    
    
def lower_bound_alignment_sampler(
    lower_bound, upper_bound,
    max_n_training_examples,
    tokenizer,
    min_lower_bound=0.00, max_upper_bound=10.00,
    source_lower_bound=None, source_upper_bound=None,
):
    instruction = f"Please say yes only if it costs between {lower_bound} and {upper_bound} dollars, otherwise no."

    all_base_input_ids = []
    all_source_input_ids = []
    all_ctf_output_ids = [] # this one does not have input ids, etc..
    all_intervention_ids = []
    dedup_set = set([])
    
    while len(all_base_input_ids) < max_n_training_examples:
        
        base_region = random.choice([1,2,3])
        source_region = random.choice([1,2,3])
        
        ctf_label = None
        if base_region == 1 and source_region >= 2:
            ctf_label = tokenizer.convert_tokens_to_ids("Yes")
        elif base_region == 2 and source_region == 2:
            ctf_label = tokenizer.convert_tokens_to_ids("Yes")
        else:
            ctf_label = tokenizer.convert_tokens_to_ids("No")
        
        base = sample_number_by_region(
            base_region, lower_bound, upper_bound,
            min_lower_bound, max_upper_bound,
            excludes=[]
        )
        
        source = sample_number_by_region(
            source_region, lower_bound, upper_bound,
            min_lower_bound, max_upper_bound,
            excludes=[base]
        )
        
        if (base, source) in dedup_set:
            continue
        dedup_set.add((base, source))
        
        base = "%.2f dollars" % base
        source = "%.2f dollars" % source
        
        # print(base, source, ctf_label)
        
        base_alpaca_prompt = alpaca_prompt_template % (instruction, base)
        source_alpaca_prompt = alpaca_prompt_template % (instruction, source)
        
        base_input_ids = tokenizer(base_alpaca_prompt, return_tensors="pt").input_ids[0]
        source_input_ids = tokenizer(source_alpaca_prompt, return_tensors="pt").input_ids[0]
        
        ctf_output_ids = (torch.ones(base_input_ids.shape[0])*-100).long().tolist()
        ctf_output_ids[-1] = ctf_label
        base_input_ids = base_input_ids.tolist()
        source_input_ids = source_input_ids.tolist()
            
        all_base_input_ids += [base_input_ids]
        all_source_input_ids += [source_input_ids]
        all_ctf_output_ids += [ctf_output_ids]
        all_intervention_ids += [0]
    
    return all_base_input_ids, all_source_input_ids, all_ctf_output_ids, all_intervention_ids

def higher_bound_alignment_sampler(
    lower_bound, upper_bound,
    max_n_training_examples,
    tokenizer,
    min_lower_bound=0.00, max_upper_bound=10.00,
    source_lower_bound=None, source_upper_bound=None,
):
    instruction = f"Please say yes only if it costs between {lower_bound} and {upper_bound} dollars, otherwise no."

    all_base_input_ids = []
    all_source_input_ids = []
    all_ctf_output_ids = [] # this one does not have input ids, etc..
    all_intervention_ids = []
    dedup_set = set([])
    
    while len(all_base_input_ids) < max_n_training_examples:
        
        base_region = random.choice([1,2,3])
        source_region = random.choice([1,2,3])
        
        ctf_label = None
        if base_region == 3 and source_region <= 2:
            ctf_label = tokenizer.convert_tokens_to_ids("Yes")
        elif base_region == 2 and source_region == 2:
            ctf_label = tokenizer.convert_tokens_to_ids("Yes")
        else:
            ctf_label = tokenizer.convert_tokens_to_ids("No")
        
        base = sample_number_by_region(
            base_region, lower_bound, upper_bound,
            min_lower_bound, max_upper_bound,
            excludes=[]
        )
        
        source = sample_number_by_region(
            source_region, lower_bound, upper_bound,
            min_lower_bound, max_upper_bound,
            excludes=[base]
        )
        
        if (base, source) in dedup_set:
            continue
        dedup_set.add((base, source))
        
        base = "%.2f dollars" % base
        source = "%.2f dollars" % source
        
        # print(base, source, ctf_label)
        
        base_alpaca_prompt = alpaca_prompt_template % (instruction, base)
        source_alpaca_prompt = alpaca_prompt_template % (instruction, source)
        
        base_input_ids = tokenizer(base_alpaca_prompt, return_tensors="pt").input_ids[0]
        source_input_ids = tokenizer(source_alpaca_prompt, return_tensors="pt").input_ids[0]
        
        ctf_output_ids = (torch.ones(base_input_ids.shape[0])*-100).long().tolist()
        ctf_output_ids[-1] = ctf_label
        base_input_ids = base_input_ids.tolist()
        source_input_ids = source_input_ids.tolist()
            
        all_base_input_ids += [base_input_ids]
        all_source_input_ids += [source_input_ids]
        all_ctf_output_ids += [ctf_output_ids]
        all_intervention_ids += [0]
    
    return all_base_input_ids, all_source_input_ids, all_ctf_output_ids, all_intervention_ids

def both_bound_alignment_sampler(
    lower_bound, upper_bound,
    max_n_training_examples,
    tokenizer,
    min_lower_bound=0.00, max_upper_bound=10.00,
    source_lower_bound=None, source_upper_bound=None,
):
    instruction = f"Please say yes only if it costs between {lower_bound} and {upper_bound} dollars, otherwise no."

    all_base_input_ids = []
    all_source_input_ids = []
    all_ctf_output_ids = [] # this one does not have input ids, etc..
    all_intervention_ids = []
    dedup_set = set([])
    
    while len(all_base_input_ids) < max_n_training_examples:
        
        if random.choice([0,1]) == 0:
        
            base_region = random.choice([1,2,3])
            source_region = random.choice([1,2,3])

            ctf_label = None
            if base_region == 1 and source_region >= 2:
                ctf_label = tokenizer.convert_tokens_to_ids("Yes")
            elif base_region == 2 and source_region == 2:
                ctf_label = tokenizer.convert_tokens_to_ids("Yes")
            else:
                ctf_label = tokenizer.convert_tokens_to_ids("No")
            
            base = sample_number_by_region(
                base_region, lower_bound, upper_bound,
                min_lower_bound, max_upper_bound,
                excludes=[]
            )

            source = sample_number_by_region(
                source_region, lower_bound, upper_bound,
                min_lower_bound, max_upper_bound,
                excludes=[base]
            )

            if (base, source) in dedup_set:
                continue
            dedup_set.add((base, source))

            base = "%.2f dollars" % base
            source = "%.2f dollars" % source

            # print(base, source, ctf_label)

            base_alpaca_prompt = alpaca_prompt_template % (instruction, base)
            source_alpaca_prompt = alpaca_prompt_template % (instruction, source)

            base_input_ids = tokenizer(base_alpaca_prompt, return_tensors="pt").input_ids[0]
            source_input_ids = tokenizer(source_alpaca_prompt, return_tensors="pt").input_ids[0]

            ctf_output_ids = (torch.ones(base_input_ids.shape[0])*-100).long().tolist()
            ctf_output_ids[-1] = ctf_label
            base_input_ids = base_input_ids.tolist()
            source_input_ids = source_input_ids.tolist()

            all_base_input_ids += [base_input_ids]
            all_source_input_ids += [source_input_ids]
            all_ctf_output_ids += [ctf_output_ids]
            all_intervention_ids += [0]
            
        else:
            
            base_region = random.choice([1,2,3])
            source_region = random.choice([1,2,3])
            
            ctf_label = None
            if base_region == 3 and source_region <= 2:
                ctf_label = tokenizer.convert_tokens_to_ids("Yes")
            elif base_region == 2 and source_region == 2:
                ctf_label = tokenizer.convert_tokens_to_ids("Yes")
            else:
                ctf_label = tokenizer.convert_tokens_to_ids("No")
            
            
            base = sample_number_by_region(
                base_region, lower_bound, upper_bound,
                min_lower_bound, max_upper_bound,
                excludes=[]
            )

            source = sample_number_by_region(
                source_region, lower_bound, upper_bound,
                min_lower_bound, max_upper_bound,
                excludes=[base]
            )

            if (base, source) in dedup_set:
                continue
            dedup_set.add((base, source))

            base = "%.2f dollars" % base
            source = "%.2f dollars" % source

            # print(base, source, ctf_label)

            base_alpaca_prompt = alpaca_prompt_template % (instruction, base)
            source_alpaca_prompt = alpaca_prompt_template % (instruction, source)

            base_input_ids = tokenizer(base_alpaca_prompt, return_tensors="pt").input_ids[0]
            source_input_ids = tokenizer(source_alpaca_prompt, return_tensors="pt").input_ids[0]

            ctf_output_ids = (torch.ones(base_input_ids.shape[0])*-100).long().tolist()
            ctf_output_ids[-1] = ctf_label
            base_input_ids = base_input_ids.tolist()
            source_input_ids = source_input_ids.tolist()

            all_base_input_ids += [base_input_ids]
            all_source_input_ids += [source_input_ids]
            all_ctf_output_ids += [ctf_output_ids]
            all_intervention_ids += [1]

    return all_base_input_ids, all_source_input_ids, all_ctf_output_ids, all_intervention_ids


def factual_sampler(
    lower_bound, upper_bound,
    tokenizer,
    min_lower_bound=0.00, max_upper_bound=10.00,
):
    instruction = f"Please say yes only if it costs between {lower_bound} and {upper_bound} dollars, otherwise no."

    all_input_ids = []
    all_output_ids = [] # this one does not have input ids, etc..
    
    result = []
    current = min_lower_bound
    step = 0.01
    raw_amounts = []
    while current <= max_upper_bound:
        amount = round(current, 2)
        if amount >= lower_bound and amount <= upper_bound:
            label = tokenizer.convert_tokens_to_ids("Yes")
        else:
            label = tokenizer.convert_tokens_to_ids("No")
            
        amount = "%.2f dollars" % amount
        
        alpaca_prompt = alpaca_prompt_template % (instruction, amount)
        input_ids = tokenizer(alpaca_prompt, return_tensors="pt").input_ids[0]
        output_ids = (torch.ones(input_ids.shape[0])*-100).long().tolist()
        output_ids[-1] = label
        input_ids = input_ids.tolist()
        
        all_input_ids += [input_ids]
        all_output_ids += [output_ids]
        raw_amounts += [current]
        
        current += step

    return all_input_ids, all_output_ids, raw_amounts

def prepare_dataset(data_args, tokenizer):
    lower_bound = float(data_args.task_config.split(";")[0])
    upper_bound = float(data_args.task_config.split(";")[1])
    min_lower_bound = float(data_args.task_config.split(";")[2])
    max_upper_bound = float(data_args.task_config.split(";")[3])
    step_size = 0.01

    logger.info(
        f"""
        Task Info:
        name = {data_args.task_name}
        lower_bound = {lower_bound}
        upper_bound = {upper_bound}
        min_lower_bound = {min_lower_bound}
        max_upper_bound = {max_upper_bound}
        """
    )

    raw_data = both_bound_alignment_sampler(
        lower_bound, upper_bound,
        data_args.max_train_samples+data_args.max_eval_samples,
        tokenizer,
        min_lower_bound=min_lower_bound, 
        max_upper_bound=max_upper_bound,
    )
    raw_train = (
        raw_data[0][:data_args.max_train_samples], 
        raw_data[1][:data_args.max_train_samples], 
        raw_data[2][:data_args.max_train_samples],
        raw_data[3][:data_args.max_train_samples]
    )
    raw_eval = (
        raw_data[0][data_args.max_train_samples:], 
        raw_data[1][data_args.max_train_samples:], 
        raw_data[2][data_args.max_train_samples:],
        raw_data[3][data_args.max_train_samples:]
    )
    train_dataset = Dataset.from_dict(
        {
            "input_ids": raw_train[0], 
            "source_input_ids": raw_train[1],
            "labels": raw_train[2],
            "intervention_ids": raw_train[3],
        }
    ).with_format("torch")
    eval_dataset = Dataset.from_dict(
        {
            "input_ids": raw_eval[0], 
            "source_input_ids": raw_eval[1],
            "labels": raw_eval[2],
            "intervention_ids": raw_eval[3],
        }
    ).with_format("torch")
    
    return train_dataset, eval_dataset

def prepare_dataloader(args, tokenizer):
    lower_bound = float(args.task_config.split(";")[0])
    upper_bound = float(args.task_config.split(";")[1])
    min_lower_bound = float(args.task_config.split(";")[2])
    max_upper_bound = float(args.task_config.split(";")[3])
    step_size = 0.01
    prealign_batch_size = args.eval_batch_size

    logger.info(
        f"""
        Task Info:
        name = {args.task_name}
        lower_bound = {lower_bound}
        upper_bound = {upper_bound}
        min_lower_bound = {min_lower_bound}
        max_upper_bound = {max_upper_bound}
        """
    )

    raw_prealign = factual_sampler(
        lower_bound, upper_bound,
        tokenizer,
        min_lower_bound=min_lower_bound, 
        max_upper_bound=max_upper_bound,
    )
    prealign_dataset = Dataset.from_dict(
        {
            "input_ids": raw_prealign[0], 
            "labels": raw_prealign[1],
        }
    ).with_format("torch")
    prealign_dataloader = DataLoader(
        prealign_dataset, batch_size=prealign_batch_size
    )
    
    if args.task_name == "cost_no_type_lower":
        sampler_func = lower_bound_alignment_sampler
    elif args.task_name == "cost_no_type_higher":
        sampler_func = higher_bound_alignment_sampler
    elif args.task_name == "cost_no_type":
        sampler_func = both_bound_alignment_sampler
    else:
        assert False
    
    raw_data = sampler_func(
        lower_bound, upper_bound,
        args.n_training_examples+args.n_eval_examples,
        tokenizer,
        min_lower_bound=min_lower_bound, 
        max_upper_bound=max_upper_bound,
    )
    raw_train = (
        raw_data[0][:args.n_training_examples], 
        raw_data[1][:args.n_training_examples], 
        raw_data[2][:args.n_training_examples],
        raw_data[3][:args.n_training_examples]
    )
    raw_eval = (
        raw_data[0][args.n_training_examples:], 
        raw_data[1][args.n_training_examples:], 
        raw_data[2][args.n_training_examples:],
        raw_data[3][args.n_training_examples:]
    )
    train_dataset = Dataset.from_dict(
        {
            "input_ids": raw_train[0], 
            "source_input_ids": raw_train[1],
            "labels": raw_train[2],
            "intervention_ids": raw_train[3],
        }
    ).with_format("torch")
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size,
    )
    eval_dataset = Dataset.from_dict(
        {
            "input_ids": raw_eval[0], 
            "source_input_ids": raw_eval[1],
            "labels": raw_eval[2],
            "intervention_ids": raw_eval[3],
        }
    ).with_format("torch")
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.eval_batch_size,
    )
    
    return train_dataloader, eval_dataloader

class AlpacaAligner(object):
    def __init__(
        self, model,
        is_master,
        logger,
        args,
        lr=5e-5,
        apex_enable=False,
        n_gpu=1,
        gpu_id=0,
        early_stopping=5,
        do_statistic=False,
        model_name="",
        intervention_config=None,
        device="cuda"
    ):
        self.model = model
        num_params = count_parameters(model)
        logger.info(f'Number of Alpaca-7B model params: {num_params}') 
        self.is_master = is_master
        self.logger = logger
        self.is_wandb = args.is_wandb
        self.model_name = model_name
        
        self.lr = lr
        self.n_gpu = n_gpu
        self.device = device
        
        self.early_stopping = early_stopping
        self.intervention_config = intervention_config
        
        if args.is_wandb and is_master:
            import wandb
            run = wandb.init(
                project="ToM-DAS-Alpaca-7B", 
                entity="wuzhengx",
                name=model_name,
            )
            wandb.config.update(args)
    
    def save_model(self, model_name):
        if self.n_gpu > 1:
            torch.save({
                'rotate_layer': self.model.module.model.rotate_layer.state_dict(),
                'intervention_boundaries': self.model.module.model.intervention_boundaries,
                'temperature': self.model.module.model.temperature
            }, os.path.join(output_dir, model_name))
        else:
            torch.save({
                'rotate_layer': self.model.model.rotate_layer.state_dict(),
                'intervention_boundaries': self.model.model.intervention_boundaries,
                'temperature': self.model.model.temperature
                
            }, os.path.join(output_dir, model_name))
    
    def train(
        self, train_dataloader, dev_dataloader,
        optimizer, scheduler, output_dir,
        log_step, valid_steps, epochs, 
        gradient_accumulation_steps,
    ):
        if self.is_master and not self.is_wandb:
            log_train = open(os.path.join(output_dir, 'train_log.txt'), 'w', buffering=1)
            log_eval = open(os.path.join(output_dir, 'eval_log.txt'), 'w', buffering=1)
            print('step,loss,accuracy', file=log_train)
            print('step,accuracy', file=log_eval)
            log_train.close()
            log_eval.close()
        
        # okay, have to honest, not sure whether we do train mode align or eval align;
        # i guess it is good to try both, but ... only trying train here and move on.
        self.model.train()
        train_iterator = trange(
            0, int(epochs), desc="Epoch"
        )
        total_step = 0
        total_log_step = 0
        best_eval_acc = -1
        target_total_step = len(train_dataloader) * int(epochs)
        temperature_start = 50.0
        temperature_end = 0.1
        temperature_schedule = torch.linspace(temperature_start, temperature_end, target_total_step)
        self.model.model.temperature.data = temperature_schedule[total_step]
        
        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc=f"Epoch: {epoch}", position=0, leave=True)
            for step, inputs in enumerate(epoch_iterator):
                
                
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                
                # aligning forward!
                source_hidden_states = self.model(
                   input_ids=inputs['source_input_ids'],
                   output_rotated_hidden_states_only=True
                ).rotated_hidden_states
                
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    source_hidden_states=source_hidden_states,
                    intervention_ids=inputs['intervention_ids'],
                    labels=inputs['labels']
                )
                
                loss = outputs.loss.mean() if self.n_gpu > 1 else outputs.loss
                
                actual_test_labels = inputs['labels'][:, -1]
                pred_test_labels = torch.argmax(outputs.logits[:, -1], dim=-1)
                correct_labels = (actual_test_labels==pred_test_labels)
                step_accuracy = correct_labels.sum() / correct_labels.shape[0]
                step_accuracy = step_accuracy.tolist()

                if self.is_master and total_step % log_step == 0:
                    if self.is_wandb:
                        wandb.log(
                            {
                                "train/loss": loss.item(),
                                "train/step_accuracy": step_accuracy
                            },
                            step=total_step
                        )
                    else:
                        log_train = open(os.path.join(output_dir, 'train_log.txt'), 'a', buffering=1)
                        print('{},{},{}'.format(
                                total_step, loss.item(), step_accuracy
                            ),
                            file=log_train
                        )
                        log_train.close()
                        
                    if total_step % valid_steps == 0:
                        total_count = 0
                        correct_count = 0
                        self.model.eval()
                        with torch.no_grad():
                            for step, inputs in enumerate(dev_dataloader):
                                for k, v in inputs.items():
                                    if v is not None and isinstance(v, torch.Tensor):
                                        inputs[k] = v.to(self.device)

                                # aligning forward!
                                source_hidden_states = self.model(
                                    input_ids=inputs['source_input_ids'],
                                    output_rotated_hidden_states_only=True
                                ).rotated_hidden_states
                                outputs = self.model(
                                    input_ids=inputs['input_ids'],
                                    source_hidden_states=source_hidden_states,
                                    intervention_ids=inputs['intervention_ids'],
                                    labels=inputs['labels']
                                )

                                actual_test_labels = inputs['labels'][:, -1]
                                pred_test_labels = torch.argmax(outputs.logits[:, -1], dim=-1)
                                correct_labels = (actual_test_labels==pred_test_labels)

                                total_count += len(correct_labels)
                                correct_count += correct_labels.sum().tolist()

                        current_acc = round(correct_count/total_count, 2)
                        if self.is_wandb:
                            wandb.log(
                                {
                                    "eval/accuracy": current_acc
                                },
                                step=total_step
                            )
                        else:
                            log_eval = open(os.path.join(output_dir, 'eval_log.txt'), 'a', buffering=1)
                            print('{},{}'.format(total_step, current_acc), file=log_eval)
                            log_eval.close()
                            
                        if current_acc > best_eval_acc:
                            best_eval_acc = current_acc
                            if self.is_master:
                                self.save_model('pytorch-rotate-best.bin')
                        self.model.train()

                    total_log_step += 1
                loss_str = round(loss.item(), 2)
                epoch_iterator.set_postfix({'loss': loss_str})
                
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                
                if total_step % gradient_accumulation_steps == 0:
                    if not (gradient_accumulation_steps > 1 and total_step == 0):
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        self.model.zero_grad()
                        self.model.model.temperature.data = temperature_schedule[total_step]
                    
                total_step += 1
                
        logger.info("Training is finished ...") 
        
        ###############################
        # End of training evaluation.
        if self.is_master:
            total_count = 0
            correct_count = 0
            self.model.eval()
            with torch.no_grad():
                for step, inputs in enumerate(dev_dataloader):
                    for k, v in inputs.items():
                        if v is not None and isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.device)

                    # aligning forward!
                    source_hidden_states = self.model(
                        input_ids=inputs['source_input_ids'],
                        output_rotated_hidden_states_only=True
                    ).rotated_hidden_states
                    outputs = self.model(
                        input_ids=inputs['input_ids'],
                        source_hidden_states=source_hidden_states,
                        intervention_ids=inputs['intervention_ids'],
                        labels=inputs['labels']
                    )

                    actual_test_labels = inputs['labels'][:, -1]
                    pred_test_labels = torch.argmax(outputs.logits[:, -1], dim=-1)
                    correct_labels = (actual_test_labels==pred_test_labels)

                    total_count += len(correct_labels)
                    correct_count += correct_labels.sum().tolist()

            current_acc = round(correct_count/total_count, 2)
            if self.is_wandb:
                wandb.log(
                    {
                        "eval/accuracy": current_acc
                    },
                    step=total_step
                )
                wandb.finish()
            else:
                log_eval = open(os.path.join(output_dir, 'eval_log.txt'), 'a', buffering=1)
                print('{},{}'.format(total_step, current_acc), file=log_eval)
                log_eval.close()
        ###############################
        
        if self.is_master:
            self.save_model('pytorch-rotate-last.bin')

        