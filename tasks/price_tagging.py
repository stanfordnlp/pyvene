from collections.abc import Callable
import os, random, argparse, sys, pickle, time, datasets
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm, trange
import numpy as np
import pandas as pd

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from datasets import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from .task_base import TaskBase

from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger('transformers')


def llama_prompt_fn(amount: float, lower_bound: float, upper_bound: float):
    alpaca_prompt_template = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
'''
    lower_bound_str = '%.2f' % lower_bound
    upper_bound_str = '%.2f' % upper_bound
    instruction = f'Please say yes only if it costs between {lower_bound_str} and {upper_bound_str} dollars, otherwise no.'
    amount_str = '%.2f dollars' % amount
    return alpaca_prompt_template.format(instruction=instruction,
                                         input=amount_str)


def t5_prompt_fn(amount: float, lower_bound: float, upper_bound: float):
    t5_fmt = 'Answer the following yes/no question:\n\nDoes the number {amount} fall in the range between {lower_bound} and {upper_bound}?\n'
    lower_bound_str = '%.2f' % lower_bound
    upper_bound_str = '%.2f' % upper_bound
    amount_str = '%.2f dollars' % amount
    return t5_fmt.format(amount=amount_str,
                         lower_bound=lower_bound_str,
                         upper_bound=upper_bound_str)


class PriceTaggingTask(TaskBase):

    def __init__(self, prompt_fn: Callable = llama_prompt_fn):
        self.prompt_fn = prompt_fn

    def price_tagging_game_config_sampler(self, amount, lower_bound,
                                          bound_width):
        if bound_width == None:
            bound_width_sample = round(random.uniform(2.50, 7.50), 2)
        else:
            bound_width_sample = bound_width
        if lower_bound == None:
            lower_bound_sample = round(
                random.uniform(0.05, 9.95 - bound_width_sample), 2)
            # left a little room to cover corner cases.
        else:
            lower_bound_sample = lower_bound
        upper_bound_sample = bound_width_sample + lower_bound_sample
        if amount == None:
            amount_sample = round(random.uniform(0.01, 9.99), 2)
        else:
            amount_sample = amount

        return lower_bound_sample, upper_bound_sample, amount_sample

    def price_tagging_game_example_sampler(
        self,
        tokenizer,
        amount,
        lower_bound,
        bound_width,
    ):
        lower_bound_sample, upper_bound_sample, amount_sample = self.price_tagging_game_config_sampler(
            amount, lower_bound, bound_width)
        if amount_sample >= lower_bound_sample and amount_sample <= upper_bound_sample:
            label = tokenizer.convert_tokens_to_ids('Yes')
        else:
            label = tokenizer.convert_tokens_to_ids('No')

        prompt = self.prompt_fn(amount_sample, lower_bound_sample,
                                upper_bound_sample)
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids[0]
        output_ids = (torch.ones(input_ids.shape[0]) * -100).long().tolist()
        output_ids[-1] = label
        input_ids = input_ids.tolist()
        # assert len(input_ids) == 82

        return input_ids, output_ids

    def price_tagging_game_example_sampler_with_info(
        self,
        tokenizer,
        amount,
        lower_bound,
        bound_width,
    ):
        lower_bound_sample, upper_bound_sample, amount_sample = self.price_tagging_game_config_sampler(
            amount, lower_bound, bound_width)
        if amount_sample >= lower_bound_sample and amount_sample <= upper_bound_sample:
            label = tokenizer.convert_tokens_to_ids('Yes')
        else:
            label = tokenizer.convert_tokens_to_ids('No')

        prompt = self.prompt_fn(amount_sample, lower_bound_sample,
                                upper_bound_sample)

        input_ids = tokenizer(paca_prompt, return_tensors='pt').input_ids[0]
        output_ids = (torch.ones(input_ids.shape[0]) * -100).long().tolist()
        output_ids[-1] = label
        input_ids = input_ids.tolist()
        # assert len(input_ids) == 82

        return input_ids, output_ids, (lower_bound_sample, upper_bound_sample,
                                       amount_sample)

    def factual_sampler(
        self,
        tokenizer,
        max_n_training_examples,
        game='price_tagging',
        amount=None,
        lower_bound=None,
        bound_width=None,
    ):

        all_input_ids = []
        all_output_ids = []  # this one does not have input ids, etc..
        for _ in range(max_n_training_examples):
            input_ids, output_ids = self.price_tagging_game_example_sampler(
                tokenizer, amount, lower_bound, bound_width)
            all_input_ids += [input_ids]
            all_output_ids += [output_ids]

        return all_input_ids, all_output_ids

    def sample_with_region(self, region, lower_bound_sample,
                           upper_bound_sample):
        if region == 1:
            amount_sample = round(
                random.uniform(0.01, lower_bound_sample - 0.01), 2)
        elif region == 2:
            amount_sample = round(
                random.uniform(lower_bound_sample, upper_bound_sample), 2)
        elif region == 3:
            amount_sample = round(
                random.uniform(upper_bound_sample + 0.01, 9.99), 2)
        return amount_sample

    def lower_bound_alignment_example_sampler(self,
                                              tokenizer,
                                              amount=None,
                                              lower_bound=None,
                                              bound_width=None):
        base_lower_bound_sample, base_upper_bound_sample, _ = \
            self.price_tagging_game_config_sampler(
                amount,
                lower_bound,
                bound_width
            )
        source_lower_bound_sample, source_upper_bound_sample, _ = \
            self.price_tagging_game_config_sampler(
                amount,
                lower_bound,
                bound_width
            )

        ctf_label_str = random.choice(['Yes', 'No'])
        if ctf_label_str == 'Yes':
            ctf_label = tokenizer.convert_tokens_to_ids('Yes')
            base_source_regions = [
                [1, 2],
                [1, 3],
                [2, 2],
            ]
        elif ctf_label_str == 'No':
            ctf_label = tokenizer.convert_tokens_to_ids('No')
            base_source_regions = [[1, 1], [2, 1], [2, 3], [3, 1], [3, 2],
                                   [3, 3]]
        base_source_region = random.choice(base_source_regions)
        base_region = base_source_region[0]
        source_region = base_source_region[1]

        base_amount_sample = self.sample_with_region(base_region,
                                                     base_lower_bound_sample,
                                                     base_upper_bound_sample)
        source_amount_sample = self.sample_with_region(
            source_region, source_lower_bound_sample,
            source_upper_bound_sample)

        return base_lower_bound_sample, base_upper_bound_sample, \
            source_lower_bound_sample, source_upper_bound_sample, \
            base_amount_sample, source_amount_sample, ctf_label, ctf_label_str

    def upper_bound_alignment_example_sampler(self,
                                              tokenizer,
                                              amount=None,
                                              lower_bound=None,
                                              bound_width=None):
        base_lower_bound_sample, base_upper_bound_sample, base_amount_sample = \
            self.price_tagging_game_config_sampler(
                amount,
                lower_bound,
                bound_width
            )
        source_lower_bound_sample, source_upper_bound_sample, source_amount_sample = \
            self.price_tagging_game_config_sampler(
                amount,
                lower_bound,
                bound_width
            )

        ctf_label_str = random.choice(['Yes', 'No'])
        if ctf_label_str == 'Yes':
            ctf_label = tokenizer.convert_tokens_to_ids('Yes')
            base_source_regions = [
                [3, 2],
                [3, 1],
                [2, 2],
            ]
        elif ctf_label_str == 'No':
            ctf_label = tokenizer.convert_tokens_to_ids('No')
            base_source_regions = [[1, 1], [1, 2], [1, 3], [2, 1], [2, 3],
                                   [3, 3]]
        base_source_region = random.choice(base_source_regions)
        base_region = base_source_region[0]
        source_region = base_source_region[1]

        base_amount_sample = self.sample_with_region(base_region,
                                                     base_lower_bound_sample,
                                                     base_upper_bound_sample)
        source_amount_sample = self.sample_with_region(
            source_region, source_lower_bound_sample,
            source_upper_bound_sample)

        return base_lower_bound_sample, base_upper_bound_sample, \
            source_lower_bound_sample, source_upper_bound_sample, \
            base_amount_sample, source_amount_sample, ctf_label, ctf_label_str

    def output_rep_bound_functor(self,
                                 tokenizer,
                                 amount=None,
                                 lower_bound=None,
                                 bound_width=None):
        # Get the base examples
        base_lower = round(random.uniform(0.05, 9.95), 2)
        base_upper = round(random.uniform(0.05, 9.95), 2)
        base_amount = round(random.uniform(0.05, 9.95), 2)
        ctf_label_str = random.choice(['Yes', 'No'])
        source_samples = [
            round(random.uniform(0.05, 9.95), 2) for _ in range(3)
        ]
        source_samples.sort()
        if ctf_label_str == 'Yes':
            source_lower = source_samples[0]
            source_amount = source_samples[1]
            source_upper = source_samples[2]
        else:
            if random.random() < 0.5:
                source_amount = source_samples[0]
                source_lower = source_samples[1]
                source_upper = source_samples[2]
            else:
                source_lower = source_samples[0]
                source_upper = source_samples[1]
                source_amount = source_samples[2]
        ctf_label = tokenizer.convert_tokens_to_ids(ctf_label_str)
        return base_lower, base_upper, source_lower, source_upper, base_amount, source_amount, ctf_label, ctf_label_str

    def bound_alignment_sampler(
        self,
        tokenizer,
        max_n_training_examples,
        bound_functors,
        amount=None,
        lower_bound=None,
        bound_width=None,
    ):
        all_base_input_ids = []
        all_source_input_ids = []
        all_ctf_output_ids = []  # this one does not have input ids, etc..
        all_intervention_ids = []

        for _ in range(max_n_training_examples):
            bound_functor = random.choice(bound_functors)
            base_lower_bound_sample, base_upper_bound_sample, \
                source_lower_bound_sample, source_upper_bound_sample, \
                base_amount_sample, source_amount_sample, \
                ctf_label, ctf_label_str = bound_functor(
                    tokenizer,
                    amount,
                    lower_bound,
                    bound_width,
                )

            base_prompt = self.prompt_fn(base_amount_sample,
                                         base_lower_bound_sample,
                                         base_upper_bound_sample)
            source_prompt = self.prompt_fn(source_amount_sample,
                                           source_lower_bound_sample,
                                           source_upper_bound_sample)

            base_input_ids = tokenizer(base_prompt,
                                       return_tensors='pt').input_ids[0]
            source_input_ids = tokenizer(source_prompt,
                                         return_tensors='pt').input_ids[0]
            base_input_ids = base_input_ids.tolist()
            source_input_ids = source_input_ids.tolist()
            ctf_output_ids = (torch.ones(len(base_input_ids)) *
                              -100).long().tolist()
            ctf_output_ids[-1] = ctf_label
            intervention_id = 0 if bound_functor == bound_functors[0] else 1

            all_base_input_ids += [base_input_ids]
            all_source_input_ids += [source_input_ids]

            all_ctf_output_ids += [ctf_output_ids]
            all_intervention_ids += [intervention_id]

            # assert len(base_input_ids) == 82
            # assert len(source_input_ids) == 82

        return all_base_input_ids, all_source_input_ids, all_ctf_output_ids, all_intervention_ids

    def midpoint_alignment_sampler(
        self,
        tokenizer,
        max_n_training_examples,
        amount=None,
        lower_bound=None,
        bound_width=None,
    ):

        all_base_input_ids = []
        all_source_input_ids = []
        all_ctf_output_ids = []  # this one does not have input ids, etc..
        all_intervention_ids = []

        for _ in range(max_n_training_examples):

            base_lower_bound_sample, base_upper_bound_sample, base_amount_sample = \
                self.price_tagging_game_config_sampler(
                    amount,
                    lower_bound,
                    bound_width
                )
            source_lower_bound_sample, source_upper_bound_sample, source_amount_sample = \
                self.price_tagging_game_config_sampler(
                    amount,
                    lower_bound,
                    bound_width
                )
            ctf_label = None
            ctf_label_str = None
            source_mid_point = (source_lower_bound_sample +
                                source_upper_bound_sample) / 2.0
            base_half = 0.5 * abs(base_upper_bound_sample -
                                  base_lower_bound_sample)
            ctf_mid_diff = abs(base_amount_sample - source_mid_point)
            if ctf_mid_diff <= base_half:
                ctf_label = tokenizer.convert_tokens_to_ids('Yes')
                ctf_label_str = 'Yes'
            else:
                ctf_label = tokenizer.convert_tokens_to_ids('No')
                ctf_label_str = 'No'

            # print(f'base: [{base_lower_bound_str}, {base_upper_bound_str}], {base_amount_str}')
            # print(f'source: [{source_lower_bound_str}, {source_upper_bound_str}], {source_amount_str}')
            # print(f'ctf label: {ctf_label_str}')

            base_prompt = self.prompt_fn(base_amount_sample,
                                         base_lower_bound_sample,
                                         base_upper_bound_sample)
            source_prompt = self.prompt_fn(source_amount_sample,
                                           source_lower_bound_sample,
                                           source_upper_bound_sample)

            base_input_ids = tokenizer(base_prompt,
                                       return_tensors='pt').input_ids[0]
            source_input_ids = tokenizer(source_prompt,
                                         return_tensors='pt').input_ids[0]
            base_input_ids = base_input_ids.tolist()
            source_input_ids = source_input_ids.tolist()
            ctf_output_ids = (torch.ones(len(base_input_ids)) *
                              -100).long().tolist()
            ctf_output_ids[-1] = ctf_label

            all_base_input_ids += [base_input_ids]
            all_source_input_ids += [source_input_ids]
            all_ctf_output_ids += [ctf_output_ids]
            all_intervention_ids += [0]
            # assert len(base_input_ids) == 82
            # assert len(source_input_ids) == 82

        return all_base_input_ids, all_source_input_ids, all_ctf_output_ids, all_intervention_ids

    def bracket_alignment_sampler(
        self,
        tokenizer,
        max_n_training_examples,
        amount=None,
        lower_bound=None,
        bound_width=None,
    ):

        all_base_input_ids = []
        all_source_input_ids = []
        all_ctf_output_ids = []  # this one does not have input ids, etc..
        all_intervention_ids = []

        for _ in range(max_n_training_examples):

            base_lower_bound_sample, base_upper_bound_sample, base_amount_sample = \
                self.price_tagging_game_config_sampler(
                    amount,
                    lower_bound,
                    bound_width
                )
            source_lower_bound_sample, source_upper_bound_sample, source_amount_sample = \
                self.price_tagging_game_config_sampler(
                    amount,
                    lower_bound,
                    bound_width
                )
            ctf_label = None
            ctf_label_str = None
            if base_amount_sample <= source_upper_bound_sample and base_amount_sample >= source_lower_bound_sample:
                ctf_label = tokenizer.convert_tokens_to_ids('Yes')
                ctf_label_str = 'Yes'
            else:
                ctf_label = tokenizer.convert_tokens_to_ids('No')
                ctf_label_str = 'No'

            base_prompt = self.prompt_fn(base_amount_sample,
                                         base_lower_bound_sample,
                                         base_upper_bound_sample)
            source_prompt = self.prompt_fn(source_amount_sample,
                                           source_lower_bound_sample,
                                           source_upper_bound_sample)

            base_input_ids = tokenizer(base_prompt,
                                       return_tensors='pt').input_ids[0]
            source_input_ids = tokenizer(source_prompt,
                                         return_tensors='pt').input_ids[0]
            base_input_ids = base_input_ids.tolist()
            source_input_ids = source_input_ids.tolist()
            ctf_output_ids = (torch.ones(len(base_input_ids)) *
                              -100).long().tolist()
            ctf_output_ids[-1] = ctf_label

            all_base_input_ids += [base_input_ids]
            all_source_input_ids += [source_input_ids]
            all_ctf_output_ids += [ctf_output_ids]
            all_intervention_ids += [0]
            # assert len(base_input_ids) == 82
            # assert len(source_input_ids) == 82

        return all_base_input_ids, all_source_input_ids, all_ctf_output_ids, all_intervention_ids

    def prepare_dataloader(self, tokenizer, **kwargs):
        '''
        Expected kwargs:
        train_batch_size
        eval_batch_size
        task_name
        n_train
        n_eval
        '''
        prealign_batch_size = kwargs['eval_batch_size']
        task_name = kwargs['task_name']
        n_train = kwargs['n_training_examples']
        n_eval = kwargs['n_eval_examples']
        logger.info(f'''
            Task Info:
            name = {task_name}
            ''')
        raw_prealign = self.factual_sampler(
            tokenizer,
            n_eval,
            game=task_name,
        )
        prealign_dataset = Dataset.from_dict({
            'input_ids':
            raw_prealign[0],
            'labels':
            raw_prealign[1],
            'output_only_labels': [output[-1:] for output in raw_prealign[1]]
        }).with_format('torch')
        prealign_dataloader = DataLoader(prealign_dataset,
                                         batch_size=prealign_batch_size)

        if task_name == 'price_tagging_lb':
            raw_data = self.bound_alignment_sampler(
                tokenizer, n_train + n_eval + 1000,
                [self.lower_bound_alignment_example_sampler])
        elif task_name == 'price_tagging_ub':
            raw_data = self.bound_alignment_sampler(
                tokenizer, n_train + n_eval + 1000,
                [self.upper_bound_alignment_example_sampler])
        elif task_name == 'price_tagging_lub':
            raw_data = self.bound_alignment_sampler(
                tokenizer, n_train + n_eval + 1000, [
                    self.lower_bound_alignment_example_sampler,
                    self.upper_bound_alignment_example_sampler
                ])
        elif task_name == 'price_tagging_mid_diff':
            raw_data = self.midpoint_alignment_sampler(
                tokenizer,
                n_train + n_eval + 1000,
            )
        elif task_name == 'price_tagging_bracket':
            raw_data = self.bracket_alignment_sampler(
                tokenizer,
                n_train + n_eval + 1000,
            )
        elif task_name == 'price_tagging_fixed':
            raw_data = self.bound_alignment_sampler(
                tokenizer,
                n_train + n_eval + 1000,
                [self.lower_bound_alignment_example_sampler],
                amount=None,
                lower_bound=5.49,
                bound_width=3.00,
            )
        elif 'output_rep' in task_name:
            raw_data = self.bound_alignment_sampler(
                tokenizer,
                n_train + n_eval + 1000,
                [self.output_rep_bound_functor],
            )

        raw_train = (raw_data[0][:n_train], raw_data[1][:n_train],
                     raw_data[2][:n_train], raw_data[3][:n_train])
        raw_eval = (raw_data[0][n_train:n_train + n_eval],
                    raw_data[1][n_train:n_train + n_eval],
                    raw_data[2][n_train:n_train + n_eval],
                    raw_data[3][n_train:n_train + n_eval])
        raw_test = (raw_data[0][n_train + n_eval:],
                    raw_data[1][n_train + n_eval:],
                    raw_data[2][n_train + n_eval:],
                    raw_data[3][n_train + n_eval:])
        train_dataset = Dataset.from_dict({
            'input_ids':
            raw_train[0],
            'source_input_ids':
            raw_train[1],
            'labels':
            raw_train[2],
            'output_only_labels': [output[-1:] for output in raw_train[2]],
            'intervention_ids':
            raw_train[3],
        }).with_format('torch')
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=kwargs['train_batch_size'],
        )
        eval_dataset = Dataset.from_dict({
            'input_ids':
            raw_eval[0],
            'source_input_ids':
            raw_eval[1],
            'labels':
            raw_eval[2],
            'output_only_labels': [output[-1:] for output in raw_eval[2]],
            'intervention_ids':
            raw_eval[3],
        }).with_format('torch')
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=kwargs['eval_batch_size'],
        )
        test_dataset = Dataset.from_dict({
            'input_ids':
            raw_test[0],
            'source_input_ids':
            raw_test[1],
            'labels':
            raw_test[2],
            'output_only_labels': [output[-1:] for output in raw_test[2]],
            'intervention_ids':
            raw_test[3],
        }).with_format('torch')
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=kwargs['eval_batch_size'],
        )
        return prealign_dataloader, train_dataloader, eval_dataloader, test_dataloader

    def sample_with_region_with_triples(self, region, triples):
        return random.choice(list(triples[region]))

    def lower_bound_alignment_example_sampler_with_triples(
        self,
        tokenizer,
        triples,
    ):

        ctf_label_str = random.choice(['Yes', 'No'])
        if ctf_label_str == 'Yes':
            ctf_label = tokenizer.convert_tokens_to_ids('Yes')
            base_source_regions = [
                [1, 2],
                [1, 3],
                [2, 2],
            ]
        elif ctf_label_str == 'No':
            ctf_label = tokenizer.convert_tokens_to_ids('No')
            base_source_regions = [[1, 1], [2, 1], [2, 3], [3, 1], [3, 2],
                                   [3, 3]]
        base_source_region = random.choice(base_source_regions)
        base_region = base_source_region[0]
        source_region = base_source_region[1]

        base_triples = self.sample_with_region_with_triples(
            base_region, triples)
        source_triples = self.sample_with_region_with_triples(
            source_region, triples)

        base_lower_bound_sample, base_upper_bound_sample, base_amount_sample = \
            base_triples
        source_lower_bound_sample, source_upper_bound_sample, source_amount_sample = \
            source_triples

        return base_lower_bound_sample, base_upper_bound_sample, \
            source_lower_bound_sample, source_upper_bound_sample, \
            base_amount_sample, source_amount_sample, ctf_label, ctf_label_str

    def upper_bound_alignment_example_sampler_with_triples(
        self,
        tokenizer,
        triples,
    ):
        ctf_label_str = random.choice(['Yes', 'No'])
        if ctf_label_str == 'Yes':
            ctf_label = tokenizer.convert_tokens_to_ids('Yes')
            base_source_regions = [
                [3, 2],
                [3, 1],
                [2, 2],
            ]
        elif ctf_label_str == 'No':
            ctf_label = tokenizer.convert_tokens_to_ids('No')
            base_source_regions = [[1, 1], [1, 2], [1, 3], [2, 1], [2, 3],
                                   [3, 3]]
        base_source_region = random.choice(base_source_regions)
        base_region = base_source_region[0]
        source_region = base_source_region[1]

        base_triples = self.sample_with_region_with_triples(
            base_region, triples)
        source_triples = self.sample_with_region_with_triples(
            source_region, triples)

        base_lower_bound_sample, base_upper_bound_sample, base_amount_sample = \
            base_triples
        source_lower_bound_sample, source_upper_bound_sample, source_amount_sample = \
            source_triples

        return base_lower_bound_sample, base_upper_bound_sample, \
            source_lower_bound_sample, source_upper_bound_sample, \
            base_amount_sample, source_amount_sample, ctf_label, ctf_label_str

    def bound_alignment_sampler_with_triples(self, tokenizer,
                                             max_n_training_examples,
                                             bound_functors, triples):
        all_base_input_ids = []
        all_source_input_ids = []
        all_ctf_output_ids = []  # this one does not have input ids, etc..
        all_intervention_ids = []

        for _ in range(max_n_training_examples):
            bound_functor = random.choice(bound_functors)
            base_lower_bound_sample, base_upper_bound_sample, \
                source_lower_bound_sample, source_upper_bound_sample, \
                base_amount_sample, source_amount_sample, \
                ctf_label, ctf_label_str = bound_functor(
                    tokenizer,
                    triples
                )

            base_prompt = self.prompt_fn(base_amount_sample,
                                         base_lower_bound_sample,
                                         base_upper_bound_sample)
            source_prompt = self.prompt_fn(source_amount_sample,
                                           source_lower_bound_sample,
                                           source_upper_bound_sample)

            base_input_ids = tokenizer(base_prompt,
                                       return_tensors='pt').input_ids[0]
            source_input_ids = tokenizer(source_prompt,
                                         return_tensors='pt').input_ids[0]
            base_input_ids = base_input_ids.tolist()
            source_input_ids = source_input_ids.tolist()
            ctf_output_ids = (torch.ones(len(base_input_ids)) *
                              -100).long().tolist()
            ctf_output_ids[-1] = ctf_label
            intervention_id = 0 if bound_functor == bound_functors[0] else 1

            all_base_input_ids += [base_input_ids]
            all_source_input_ids += [source_input_ids]

            all_ctf_output_ids += [ctf_output_ids]
            all_intervention_ids += [intervention_id]

            # assert len(base_input_ids) == 82
            # assert len(source_input_ids) == 82

        return all_base_input_ids, all_source_input_ids, all_ctf_output_ids, all_intervention_ids
