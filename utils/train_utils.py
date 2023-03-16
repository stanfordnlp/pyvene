import os, random, argparse, sys, pickle, time
import torch
from transformers import AutoConfig, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from models.modelings_alignable_gpt2 import *
from logic_data.constants import *
from datasets import Dataset 
from torch.utils.data import DataLoader
import gc

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import logging
logging.basicConfig(level = logging.INFO)

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