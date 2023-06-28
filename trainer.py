import os, random, argparse, sys, pickle, time
import torch
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import Dataset 
from torch.utils.data import DataLoader
from dataclasses import dataclass, field

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger("transformers")
"""
This code is designed for alignment search
for large models, i.e., >1B parameters.

We test it out with Alpaca 7B which is based
on LLaMA 7B model, but it should be extensible
to larger models as well if computation resource
is allowed.
"""
CACHE_DIR = "../.cache/"

class Aligner(object):
    def __init__(
        self, model,
        is_master,
        logger,
        is_wandb,
        compute_metrics,
        lr=5e-5,
        apex_enable=False,
        n_gpu=1,
        gpu_id=0,
        early_stopping=5,
        do_statistic=False,
        model_name="",
        device="cuda"
    ):
        self.model = model
        num_params = count_parameters(model)
        logger.info(f'Number of aligning model params: {num_params}') 
        self.is_master = is_master
        self.logger = logger
        self.is_wandb = is_wandb
        self.model_name = model_name
        self.compute_metrics_fn = compute_metrics
        
        self.lr = lr
        self.n_gpu = n_gpu
        self.device = device
        
        self.early_stopping = early_stopping
    
    def save_model(self, output_dir, model_name):
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
    
    def prealign_eval(self, prealign_dataloader, output_dir):
        eval_labels = []
        eval_preds = []
        self.model.eval()
        with torch.no_grad():
            for step, inputs in enumerate(prealign_dataloader):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                # aligning forward!
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    labels=inputs['labels']
                )
                eval_labels += [inputs['labels']]
                eval_preds += [outputs.logits]
        eval_metrics = self.compute_metrics_fn(eval_preds, eval_labels)
        logger.info(f"[WARNING: THIS NEEDS TO BE GOOD!] prealign task accuracy: {eval_metrics['accuracy']}")
        
        if self.is_master and not self.is_wandb:
            log_prealign = open(os.path.join(output_dir, 'prealign_log.txt'), 'w', buffering=1)
            print(f"prealign_accuracy,{eval_metrics['accuracy']}", file=log_prealign)
            log_prealign.close()
        elif self.is_wandb:
            wandb.log(
                {
                    "eval/prealign_accuracy": eval_metrics['accuracy']
                },
                step=0
            )
            
    def train(
        self, train_dataloader, dev_dataloader, test_dataloader,
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
        temperature_schedule = torch.linspace(temperature_start, temperature_end, target_total_step).to(torch.bfloat16)
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
                step_accuracy = self.compute_metrics_fn([outputs.logits], [inputs['labels']])['accuracy']

                if self.is_master and total_step % log_step == 0:
                    if self.is_wandb:
                        intervention_boundaries = torch.clamp(self.model.model.intervention_boundaries, 1e-3, 1)
                        wandb.log(
                            {
                                "train/loss": loss.item(),
                                "train/step_accuracy": step_accuracy,
                                "train/temperature": self.model.model.temperature.data,
                                "train/unified_boundary": intervention_boundaries.data[0],
                                "train/unified_boundary (dummy)": intervention_boundaries.data[1],                                       
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
                        
                    if total_step != 0 and total_step % valid_steps == 0:
                        eval_labels = []
                        eval_preds = []
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

                                eval_labels += [inputs['labels']]
                                eval_preds += [outputs.logits]
                        eval_metrics = self.compute_metrics_fn(eval_preds, eval_labels)
                        
                        if self.is_wandb:
                            wandb.log(
                                {
                                    "eval/accuracy": eval_metrics['accuracy']
                                },
                                step=total_step
                            )
                        else:
                            log_eval = open(os.path.join(output_dir, 'eval_log.txt'), 'a', buffering=1)
                            print('{},{}'.format(total_step, eval_metrics['accuracy']), file=log_eval)
                            log_eval.close()
                            
                        if eval_metrics['accuracy'] > best_eval_acc:
                            best_eval_acc = eval_metrics['accuracy']
                            if self.is_master:
                                self.save_model(output_dir, 'pytorch-rotate-best.bin')
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
            self.model.eval()
            eval_labels = []
            eval_preds = []
            with torch.no_grad():
                for step, inputs in enumerate(test_dataloader):
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
                    
                    eval_labels += [inputs['labels']]
                    eval_preds += [outputs.logits]
            eval_metrics = self.compute_metrics_fn(eval_preds, eval_labels)
            
            if self.is_wandb:
                wandb.log(
                    {
                        "test/accuracy": eval_metrics['accuracy']
                    },
                    step=total_step
                )
                wandb.finish()
            else:
                log_eval = open(os.path.join(output_dir, 'eval_log.txt'), 'a', buffering=1)
                print('{},{}'.format(total_step, eval_metrics['accuracy']), file=log_eval)
                log_eval.close()
        ###############################
        
        if self.is_master:
            self.save_model(output_dir, 'pytorch-rotate-last.bin')

        