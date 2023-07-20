import wandb
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

    def __init__(self,
                 model,
                 tokenizer,
                 is_master,
                 logger,
                 is_wandb,
                 lr=5e-5,
                 apex_enable=False,
                 n_gpu=1,
                 gpu_id=0,
                 early_stopping=5,
                 do_statistic=False,
                 model_name="",
                 model_type="",
                 run_name="",
                 task_name="",
                 device="cuda",
                 token_metric_fns=[],
                 decoded_metric_fns=[]):
        self.model = model
        num_params = count_parameters(model)
        logger.info(f'Number of Alpaca-7B model params: {num_params}')
        self.is_master = is_master
        self.logger = logger
        self.is_wandb = is_wandb
        self.run_name = run_name
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.lr = lr
        self.n_gpu = n_gpu
        self.device = device
        self.token_metric_fns = token_metric_fns
        self.decoded_metric_fns = decoded_metric_fns

        self.early_stopping = early_stopping

    def call_model(self,
                   inputs: dict,
                   run_intervention=False,
                   compute_prediction=False,
                   **kwargs):
        """If compute_prediction=True, this will take argmax of logits. Otherwise, just return model output object."""
        attention_mask = inputs.get('attention_masks', None)
        if self.model_type == 't5':
            if run_intervention:
                source_hidden_states = self.model(
                    input_ids=inputs['input_ids'],
                    output_rotated_hidden_states_only=True,
                    attention_mask=inputs['attention_masks'],
                    labels=inputs['output_only_labels']).rotated_hidden_states

                model_outputs = self.model(
                    input_ids=inputs['input_ids'],
                    source_hidden_states=source_hidden_states,
                    intervention_ids=inputs['intervention_ids'],
                    labels=inputs['output_only_labels'])
            else:
                model_outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_masks'],
                    labels=inputs['output_only_labels'])

        else:
            if run_intervention:
                source_hidden_states = self.model(
                    input_ids=inputs['source_input_ids'],
                    output_rotated_hidden_states_only=True
                ).rotated_hidden_states

                model_outputs = self.model(
                    input_ids=inputs['input_ids'],
                    source_hidden_states=source_hidden_states,
                    intervention_ids=inputs['intervention_ids'],
                    labels=inputs['labels'])
            else:
                model_outputs = self.model(input_ids=inputs['input_ids'],
                                           labels=inputs['labels'])
            model_outputs = self.model(input_ids=inputs['input_ids'],
                                       labels=labels,
                                       **kwargs)
        if compute_prediction:
            return torch.argmax(model_outputs.logits[:, -1], dim=-1)

        return model_outputs

    def save_model(self, output_dir, model_name):
        if self.n_gpu > 1:
            torch.save(
                {
                    'rotate_layer':
                    self.model.module.model.rotate_layer.state_dict(),
                    'intervention_boundaries':
                    self.model.module.model.intervention_boundaries,
                    'temperature':
                    self.model.module.model.temperature
                }, os.path.join(output_dir, model_name))
        else:
            torch.save(
                {
                    'rotate_layer': self.model.get_rotation_parameters(),
                    'intervention_boundaries':
                    self.model.get_boundary_parameters(),
                    'temperature': self.model.get_temperature()
                }, os.path.join(output_dir, model_name))

    def compute_token_metrics(self, preds, labels):
        metrics = {}
        for fn in self.token_metric_fns:
            metrics.update(fn(preds, labels))
        return metrics

    def compute_decoded_metrics(self, preds, labels):
        metrics = {}
        for fn in self.decoded_metric_fns:
            metrics.update(fn(preds, labels))
        return metrics

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
                input_ids = inputs['input_ids'].tolist()
                inputs_decoded = self.tokenizer.batch_decode(
                    input_ids, skip_special_tokens=True)
                outputs = self.call_model(inputs,
                                          labels=inputs['labels'],
                                          compute_prediction=True)
                # Sometimes the model gets the case wrong, so let's decode to strings and match.

                actual_test_labels = inputs['labels'][:, -1]
                target_decoded = self.tokenizer.batch_decode(
                    actual_test_labels, skip_special_tokens=True)
                pred_decoded = self.tokenizer.batch_decode(
                    outputs, skip_special_tokens=True)
                eval_preds.extend(pred_decoded)
                eval_labels.extend(target_decoded)

        eval_metrics = self.compute_decoded_metrics(eval_preds, eval_labels)
        logger.info(
            f"[WARNING: THIS NEEDS TO BE GOOD!] prealign task metrics: {eval_metrics}"
        )

        if self.is_master and not self.is_wandb:
            log_prealign = open(os.path.join(output_dir, 'prealign_log.txt'),
                                'w',
                                buffering=1)
            print(f'prealign_accuracy,{current_acc}', file=log_prealign)
            log_prealign.close()
        elif self.is_wandb:
            wandb_dict = {f'prealign/{k}': v for k, v in eval_metrics.items()}
            wandb.log(wandb_dict, step=0)

    def train(
        self,
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        optimizer,
        scheduler,
        output_dir,
        log_step,
        valid_steps,
        epochs,
        gradient_accumulation_steps,
    ):
        if self.is_master and not self.is_wandb:
            log_train = open(os.path.join(output_dir, 'train_log.txt'),
                             'w',
                             buffering=1)
            log_eval = open(os.path.join(output_dir, 'eval_log.txt'),
                            'w',
                            buffering=1)
            print('step,loss,accuracy', file=log_train)
            print('step,accuracy', file=log_eval)
            log_train.close()
            log_eval.close()

        # okay, have to honest, not sure whether we do train mode align or eval align;
        # i guess it is good to try both, but ... only trying train here and move on.
        self.model.train()
        train_iterator = trange(0, int(epochs), desc="Epoch")
        total_step = 0
        total_log_step = 0
        best_eval_acc = -1
        target_total_step = len(train_dataloader) * int(epochs)
        temperature_start = 50.0
        temperature_end = 0.1
        temperature_schedule = torch.linspace(
            temperature_start, temperature_end,
            target_total_step).to(torch.bfloat16)
        self.model.set_temperature(temperature_schedule[total_step])

        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader,
                                  desc=f"Epoch: {epoch}",
                                  position=0,
                                  leave=True)
            for step, inputs in enumerate(epoch_iterator):

                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                # aligning forward!
                outputs = self.call_model(inputs,
                                          run_intervention=True,
                                          compute_prediction=False)

                loss = outputs.loss.mean() if self.n_gpu > 1 else outputs.loss

                actual_test_labels = inputs['labels'][:, -1]
                pred_test_labels = torch.argmax(outputs.logits[:, -1], dim=-1)

                target_decoded = self.tokenizer.batch_decode(
                    actual_test_labels, skip_special_tokens=True)
                pred_decoded = self.tokenizer.batch_decode(
                    pred_test_labels, skip_special_tokens=True)

                token_metrics = self.compute_token_metrics(
                    pred_test_labels, actual_test_labels)
                decoded_metrics = self.compute_decoded_metrics(
                    pred_decoded, target_decoded)

                if self.is_master and total_step % log_step == 0:
                    if self.is_wandb:
                        intervention_boundaries = torch.clamp(
                            self.model.get_boundary_parameters(), 1e-3, 1)
                        wandb_dict = {
                            "train/loss":
                            loss.item(),
                            "train/temperature":
                            self.model.get_temperature().data,
                            "train/boundary0":
                            intervention_boundaries.data[0],
                            "train/boundary1":
                            intervention_boundaries.data[1],
                            "train/rotate_layer_params":
                            wandb.Histogram([
                                p.detach().cpu().float().numpy()
                                for p in self.model.get_rotation_parameters()
                            ]),
                        }
                        for k, v in token_metrics.items():
                            wandb_dict[f'token_metrics/{k}'] = v
                        for k, v in decoded_metrics.items():
                            wandb_dict[f'decoded_metrics/{k}'] = v
                        wandb.log(wandb_dict, step=total_step)
                    else:
                        log_train = open(os.path.join(output_dir,
                                                      'train_log.txt'),
                                         'a',
                                         buffering=1)
                        print('{},{},{},token_metrics'.format(
                            total_step, loss.item(), token_metrics),
                              file=log_train)
                        print('{},{},{},decoded_metrics'.format(
                            total_step, loss.item(), decoded_metrics),
                              file=log_train)
                        log_train.close()

                    total_log_step += 1
                loss_str = round(loss.item(), 2)
                epoch_iterator.set_postfix({'loss': loss_str})

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                if total_step % gradient_accumulation_steps == 0:
                    if not (gradient_accumulation_steps > 1
                            and total_step == 0):
                        loss.backward()
                        wandb.log({
                            'train/rotate_layer_gradients':
                            wandb.Histogram([
                                p.grad.float().cpu().numpy()
                                for p in self.model.get_rotation_parameters()
                            ])
                        })
                        optimizer.step()
                        scheduler.step()
                        self.model.zero_grad()
                        self.model.set_temperature(
                            temperature_schedule[total_step])

                if total_step != 0 and total_step % valid_steps == 0:
                    token_labels = []
                    token_preds = []
                    eval_labels = []
                    eval_preds = []
                    self.model.eval()
                    with torch.no_grad():
                        for step, inputs in enumerate(dev_dataloader):
                            for k, v in inputs.items():
                                if v is not None and isinstance(
                                        v, torch.Tensor):
                                    inputs[k] = v.to(self.device)
                            # aligning forward!
                            source_hidden_states = self.call_model(
                                inputs, output_rotated_hidden_states_only=True
                            ).rotated_hidden_states

                            outputs = self.call_model(
                                inputs,
                                source_hidden_states=source_hidden_states,
                                intervention_ids=inputs['intervention_ids'],
                                labels=inputs['labels'])

                            loss = outputs.loss.mean(
                            ) if self.n_gpu > 1 else outputs.loss

                            actual_test_labels = inputs['labels'][:, -1]
                            pred_test_labels = torch.argmax(outputs.logits[:,
                                                                           -1],
                                                            dim=-1)
                            token_labels.extend(actual_test_labels)
                            token_preds.extend(pred_test_labels)

                            target_decoded = self.tokenizer.batch_decode(
                                actual_test_labels, skip_special_tokens=True)
                            pred_decoded = self.tokenizer.batch_decode(
                                pred_test_labels, skip_special_tokens=True)
                            eval_labels.extend(target_decoded)
                            eval_preds.extend(pred_decoded)
                    decoded_metrics = self.compute_decoded_metrics(
                        eval_preds, eval_labels)
                    token_metrics = self.compute_token_metrics(
                        token_preds, token_labels)
                    if self.is_wandb:
                        wandb_dict = {}
                        for k, v in token_metrics.items():
                            wandb_dict[f'eval_token_metrics/{k}'] = v
                        for k, v in decoded_metrics.items():
                            wandb_dict[f'eval_decoded_metrics/{k}'] = v
                        wandb.log(wandb_dict, step=total_step)
                    else:
                        log_eval = open(os.path.join(output_dir,
                                                     'eval_log.txt'),
                                        'a',
                                        buffering=1)
                        print('{},{},token_metrics'.format(
                            total_step, token_metrics),
                              file=log_eval)
                        print('{},{},decoded_metrics'.format(
                            total_step, decoded_metrics),
                              file=log_eval)
                        log_eval.close()

                    current_acc = decoded_metrics['str_accuracy']
                    if current_acc > best_eval_acc:
                        best_eval_acc = current_acc
                        if self.is_master:
                            self.save_model(output_dir,
                                            'pytorch-rotate-best.bin')
                    self.model.train()

                total_step += 1

        logger.info("Training is finished ...")

        ###############################
        # End of training evaluation.
        if self.is_master:
            self.model.eval()
            eval_labels = []
            eval_preds = []
            token_labels = []
            token_preds = []
            with torch.no_grad():
                for step, inputs in enumerate(test_dataloader):
                    for k, v in inputs.items():
                        if v is not None and isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.device)
                    # aligning forward!
                    source_hidden_states = self.call_model(
                        inputs, output_rotated_hidden_states_only=True
                    ).rotated_hidden_states

                    outputs = self.call_model(
                        inputs,
                        source_hidden_states=source_hidden_states,
                        intervention_ids=inputs['intervention_ids'],
                        labels=inputs['labels'])

                    loss = outputs.loss.mean(
                    ) if self.n_gpu > 1 else outputs.loss

                    actual_test_labels = inputs['labels'][:, -1]
                    pred_test_labels = torch.argmax(outputs.logits[:, -1],
                                                    dim=-1)
                    token_labels.extend(actual_test_labels)
                    token_preds.extend(pred_test_labels)

                    target_decoded = self.tokenizer.batch_decode(
                        actual_test_labels, skip_special_tokens=True)
                    pred_decoded = self.tokenizer.batch_decode(
                        pred_test_labels, skip_special_tokens=True)
                    eval_labels.extend(target_decoded)
                    eval_preds.extend(pred_decoded)

            decoded_metrics = self.compute_decoded_metrics(
                eval_preds, eval_labels)
            token_metrics = self.compute_token_metrics(token_preds,
                                                       token_labels)
            if self.is_wandb:
                wandb_dict = {}
                for k, v in token_metrics.items():
                    wandb_dict[f'test_token_metrics/{k}'] = v
                for k, v in decoded_metrics.items():
                    wandb_dict[f'test_decoded_metrics/{k}'] = v
                wandb.log(wandb_dict, step=total_step)
                wandb.finish()
            else:
                log_eval = open(os.path.join(output_dir, 'eval_log.txt'),
                                'a',
                                buffering=1)
                print('{},{},token_metrics'.format(total_step, token_metrics),
                      file=log_eval)
                print('{},{},decoded_metrics'.format(total_step,
                                                     decoded_metrics),
                      file=log_eval)
                log_eval.close()
        ###############################

        if self.is_master:
            self.save_model(output_dir, 'pytorch-rotate-last.bin')
