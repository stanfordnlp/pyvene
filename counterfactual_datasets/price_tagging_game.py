import os, random, argparse, sys, pickle, time, datasets
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import Dataset 
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
    
alpaca_prompt_template = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
%s

### Input:
%s

### Response:
"""

from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger("transformers")

def pricing_tag_game_config_sampler(
    amount,
    lower_bound,
    bound_width
):
    if bound_width == None:
        bound_width_sample = round(random.uniform(2.50, 7.50), 2)
    else:
        bound_width_sample = bound_width
    if lower_bound == None:
        lower_bound_sample = round(random.uniform(0.05, 9.95-bound_width_sample), 2)
        # left a little room to cover corner cases.
    else:
        lower_bound_sample = lower_bound
    upper_bound_sample = bound_width_sample + lower_bound_sample
    if amount == None:
        amount_sample = round(random.uniform(0.01, 9.99), 2)
    else:
        amount_sample = amount
        
    return lower_bound_sample, upper_bound_sample, amount_sample

def pricing_tag_game_example_sampler(
    tokenizer,
    amount,
    lower_bound,
    bound_width,
):
    lower_bound_sample, upper_bound_sample, amount_sample = pricing_tag_game_config_sampler(
        amount,
        lower_bound,
        bound_width
    )
    lower_bound_str = "%.2f" % lower_bound_sample
    upper_bound_str = "%.2f" % upper_bound_sample
    if amount_sample >= float(lower_bound_str) and amount_sample <= float(upper_bound_str):
        label = tokenizer.convert_tokens_to_ids("Yes")
    else:
        label = tokenizer.convert_tokens_to_ids("No")

    amount_str = "%.2f dollars" % amount_sample
    instruction = f"Please say yes only if it costs between {lower_bound_str} and {upper_bound_str} dollars, otherwise no."
    alpaca_prompt = alpaca_prompt_template % (instruction, amount_str)
    input_ids = tokenizer(alpaca_prompt, return_tensors="pt").input_ids[0]
    output_ids = (torch.ones(input_ids.shape[0])*-100).long().tolist()
    output_ids[-1] = label
    input_ids = input_ids.tolist()
    assert len(input_ids) == 82
    
    return input_ids, output_ids
        
def pricing_tag_game_example_sampler_with_info(
    tokenizer,
    amount,
    lower_bound,
    bound_width,
):
    lower_bound_sample, upper_bound_sample, amount_sample = pricing_tag_game_config_sampler(
        amount,
        lower_bound,
        bound_width
    )
    lower_bound_str = "%.2f" % lower_bound_sample
    upper_bound_str = "%.2f" % upper_bound_sample
    if amount_sample >= float(lower_bound_str) and amount_sample <= float(upper_bound_str):
        label = tokenizer.convert_tokens_to_ids("Yes")
    else:
        label = tokenizer.convert_tokens_to_ids("No")

    amount_str = "%.2f dollars" % amount_sample
    instruction = f"Please say yes only if it costs between {lower_bound_str} and {upper_bound_str} dollars, otherwise no."
    alpaca_prompt = alpaca_prompt_template % (instruction, amount_str)
    input_ids = tokenizer(alpaca_prompt, return_tensors="pt").input_ids[0]
    output_ids = (torch.ones(input_ids.shape[0])*-100).long().tolist()
    output_ids[-1] = label
    input_ids = input_ids.tolist()
    assert len(input_ids) == 82
    
    return input_ids, output_ids, (lower_bound_sample, upper_bound_sample, amount_sample)
    
def factual_sampler(
    tokenizer,
    max_n_training_examples,
    game="pricing_tag",
    amount=None,
    lower_bound=None,
    bound_width=None,
):
    
    all_input_ids = []
    all_output_ids = [] # this one does not have input ids, etc..
    for _ in range(max_n_training_examples):
        if "pricing_tag" in game:
            input_ids, output_ids = pricing_tag_game_example_sampler(
                tokenizer,
                amount,
                lower_bound,
                bound_width
            )
        elif game == "continent_retrieval":
            pass
        all_input_ids += [input_ids]
        all_output_ids += [output_ids]
        
    return all_input_ids, all_output_ids

def sample_with_region(
    region,
    lower_bound_sample,
    upper_bound_sample
):
    if region == 1:
        amount_sample = round(random.uniform(0.01, lower_bound_sample - 0.01), 2)
    elif region == 2:
        amount_sample = round(random.uniform(lower_bound_sample, upper_bound_sample), 2)
    elif region == 3:
        amount_sample = round(random.uniform(upper_bound_sample + 0.01, 9.99), 2)
    return amount_sample
        
def lower_bound_alignment_example_sampler(
    tokenizer,
    amount=None,
    lower_bound=None,
    bound_width=None
):
    base_lower_bound_sample, base_upper_bound_sample, _ = \
        pricing_tag_game_config_sampler(
            amount,
            lower_bound,
            bound_width
        )
    source_lower_bound_sample, source_upper_bound_sample, _ = \
        pricing_tag_game_config_sampler(
            amount,
            lower_bound,
            bound_width
        )
    
    ctf_label_str = random.choice(["Yes", "No"])
    if ctf_label_str == "Yes":
        ctf_label = tokenizer.convert_tokens_to_ids("Yes")
        base_source_regions = [
            [1,2],
            [1,3],
            [2,2],
        ]
    elif ctf_label_str == "No":
        ctf_label = tokenizer.convert_tokens_to_ids("No")
        base_source_regions = [
            [1,1],
            [2,1],
            [2,3],
            [3,1],
            [3,2],
            [3,3]
        ]
    base_source_region = random.choice(base_source_regions)
    base_region = base_source_region[0]
    source_region = base_source_region[1]

    base_amount_sample = sample_with_region(
        base_region, base_lower_bound_sample, base_upper_bound_sample)
    source_amount_sample = sample_with_region(
        source_region, source_lower_bound_sample, source_upper_bound_sample)
        
    return base_lower_bound_sample, base_upper_bound_sample, \
        source_lower_bound_sample, source_upper_bound_sample, \
        base_amount_sample, source_amount_sample, ctf_label, ctf_label_str
    
def upper_bound_alignment_example_sampler(
    tokenizer,
    amount=None,
    lower_bound=None,
    bound_width=None
):
    base_lower_bound_sample, base_upper_bound_sample, base_amount_sample = \
        pricing_tag_game_config_sampler(
            amount,
            lower_bound,
            bound_width
        )
    source_lower_bound_sample, source_upper_bound_sample, source_amount_sample = \
        pricing_tag_game_config_sampler(
            amount,
            lower_bound,
            bound_width
        )
    
    ctf_label_str = random.choice(["Yes", "No"])
    if ctf_label_str == "Yes":
        ctf_label = tokenizer.convert_tokens_to_ids("Yes")
        base_source_regions = [
            [3,2],
            [3,1],
            [2,2],
        ]
    elif ctf_label_str == "No":
        ctf_label = tokenizer.convert_tokens_to_ids("No")
        base_source_regions = [
            [1,1],
            [1,2],
            [1,3],
            [2,1],
            [2,3],
            [3,3]
        ]
    base_source_region = random.choice(base_source_regions)
    base_region = base_source_region[0]
    source_region = base_source_region[1]
    
    base_amount_sample = sample_with_region(
        base_region, base_lower_bound_sample, base_upper_bound_sample)
    source_amount_sample = sample_with_region(
        source_region, source_lower_bound_sample, source_upper_bound_sample)
    
    return base_lower_bound_sample, base_upper_bound_sample, \
        source_lower_bound_sample, source_upper_bound_sample, \
        base_amount_sample, source_amount_sample, ctf_label, ctf_label_str

def bound_alignment_sampler(
    tokenizer,
    max_n_training_examples,
    bound_functors,
    amount=None,
    lower_bound=None,
    bound_width=None,
):
    all_base_input_ids = []
    all_source_input_ids = []
    all_ctf_output_ids = [] # this one does not have input ids, etc..
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

        base_amount_str = "%.2f dollars" % base_amount_sample
        source_amount_str = "%.2f dollars" % source_amount_sample
        base_lower_bound_str = "%.2f" % base_lower_bound_sample
        base_upper_bound_str = "%.2f" % base_upper_bound_sample
        source_lower_bound_str = "%.2f" % source_lower_bound_sample
        source_upper_bound_str = "%.2f" % source_upper_bound_sample
        
        # print(f"base: [{base_lower_bound_str}, {base_upper_bound_str}], {base_amount_str}")
        # print(f"source: [{source_lower_bound_str}, {source_upper_bound_str}], {source_amount_str}")
        # print(f"ctf label: {ctf_label_str}")
        
        base_instruction = f"Please say yes only if it costs between {base_lower_bound_str} and {base_upper_bound_str} dollars, otherwise no."
        source_instruction = f"Please say yes only if it costs between {source_lower_bound_str} and {source_upper_bound_str} dollars, otherwise no."
        
        base_alpaca_prompt = alpaca_prompt_template % (base_instruction, base_amount_str)
        source_alpaca_prompt = alpaca_prompt_template % (source_instruction, source_amount_str)
        
        base_input_ids = tokenizer(base_alpaca_prompt, return_tensors="pt").input_ids[0]
        source_input_ids = tokenizer(source_alpaca_prompt, return_tensors="pt").input_ids[0]
        base_input_ids = base_input_ids.tolist()
        source_input_ids = source_input_ids.tolist()
        ctf_output_ids = (torch.ones(len(base_input_ids))*-100).long().tolist()
        ctf_output_ids[-1] = ctf_label
        intervention_id = 0 if bound_functor == bound_functors[0] else 1
        
        all_base_input_ids += [base_input_ids]
        all_source_input_ids += [source_input_ids]
        
        all_ctf_output_ids += [ctf_output_ids]
        all_intervention_ids += [intervention_id]
        
        assert len(base_input_ids) == 82
        assert len(source_input_ids) == 82
        
    return all_base_input_ids, all_source_input_ids, all_ctf_output_ids, all_intervention_ids

def midpoint_alignment_sampler(
    tokenizer,
    max_n_training_examples,
    amount=None,
    lower_bound=None,
    bound_width=None,
):

    all_base_input_ids = []
    all_source_input_ids = []
    all_ctf_output_ids = [] # this one does not have input ids, etc..
    all_intervention_ids = []
    
    for _ in range(max_n_training_examples):
        
        base_lower_bound_sample, base_upper_bound_sample, base_amount_sample = \
            pricing_tag_game_config_sampler(
                amount,
                lower_bound,
                bound_width
            )
        source_lower_bound_sample, source_upper_bound_sample, source_amount_sample = \
            pricing_tag_game_config_sampler(
                amount,
                lower_bound,
                bound_width
            )
        ctf_label = None
        ctf_label_str = None
        source_mid_point = (source_lower_bound_sample+source_upper_bound_sample)/2.0
        base_half = 0.5*abs(base_upper_bound_sample-base_lower_bound_sample)
        ctf_mid_diff = abs(base_amount_sample-source_mid_point)
        if ctf_mid_diff <= base_half:
            ctf_label = tokenizer.convert_tokens_to_ids("Yes")
            ctf_label_str = "Yes"
        else:
            ctf_label = tokenizer.convert_tokens_to_ids("No")
            ctf_label_str = "No"
            
        base_amount_str = "%.2f dollars" % base_amount_sample
        source_amount_str = "%.2f dollars" % source_amount_sample
        base_lower_bound_str = "%.2f" % base_lower_bound_sample
        base_upper_bound_str = "%.2f" % base_upper_bound_sample
        source_lower_bound_str = "%.2f" % source_lower_bound_sample
        source_upper_bound_str = "%.2f" % source_upper_bound_sample
        
        # print(f"base: [{base_lower_bound_str}, {base_upper_bound_str}], {base_amount_str}")
        # print(f"source: [{source_lower_bound_str}, {source_upper_bound_str}], {source_amount_str}")
        # print(f"ctf label: {ctf_label_str}")
        
        base_instruction = f"Please say yes only if it costs between {base_lower_bound_str} and {base_upper_bound_str} dollars, otherwise no."
        source_instruction = f"Please say yes only if it costs between {source_lower_bound_str} and {source_upper_bound_str} dollars, otherwise no."
        
        base_alpaca_prompt = alpaca_prompt_template % (base_instruction, base_amount_str)
        source_alpaca_prompt = alpaca_prompt_template % (source_instruction, source_amount_str)
        
        base_input_ids = tokenizer(base_alpaca_prompt, return_tensors="pt").input_ids[0]
        source_input_ids = tokenizer(source_alpaca_prompt, return_tensors="pt").input_ids[0]
        base_input_ids = base_input_ids.tolist()
        source_input_ids = source_input_ids.tolist()
        ctf_output_ids = (torch.ones(len(base_input_ids))*-100).long().tolist()
        ctf_output_ids[-1] = ctf_label
        
        all_base_input_ids += [base_input_ids]
        all_source_input_ids += [source_input_ids]
        all_ctf_output_ids += [ctf_output_ids]
        all_intervention_ids += [0]
        assert len(base_input_ids) == 82
        assert len(source_input_ids) == 82
        
    return all_base_input_ids, all_source_input_ids, all_ctf_output_ids, all_intervention_ids

def bracket_alignment_sampler(
    tokenizer,
    max_n_training_examples,
    amount=None,
    lower_bound=None,
    bound_width=None,
):

    all_base_input_ids = []
    all_source_input_ids = []
    all_ctf_output_ids = [] # this one does not have input ids, etc..
    all_intervention_ids = []
    
    for _ in range(max_n_training_examples):
        
        base_lower_bound_sample, base_upper_bound_sample, base_amount_sample = \
            pricing_tag_game_config_sampler(
                amount,
                lower_bound,
                bound_width
            )
        source_lower_bound_sample, source_upper_bound_sample, source_amount_sample = \
            pricing_tag_game_config_sampler(
                amount,
                lower_bound,
                bound_width
            )
        ctf_label = None
        ctf_label_str = None
        if base_amount_sample <= source_upper_bound_sample and base_amount_sample >= source_lower_bound_sample:
            ctf_label = tokenizer.convert_tokens_to_ids("Yes")
            ctf_label_str = "Yes"
        else:
            ctf_label = tokenizer.convert_tokens_to_ids("No")
            ctf_label_str = "No"
            
        base_amount_str = "%.2f dollars" % base_amount_sample
        source_amount_str = "%.2f dollars" % source_amount_sample
        base_lower_bound_str = "%.2f" % base_lower_bound_sample
        base_upper_bound_str = "%.2f" % base_upper_bound_sample
        source_lower_bound_str = "%.2f" % source_lower_bound_sample
        source_upper_bound_str = "%.2f" % source_upper_bound_sample
        
        # print(f"base: [{base_lower_bound_str}, {base_upper_bound_str}], {base_amount_str}")
        # print(f"source: [{source_lower_bound_str}, {source_upper_bound_str}], {source_amount_str}")
        # print(f"ctf label: {ctf_label_str}")
        
        base_instruction = f"Please say yes only if it costs between {base_lower_bound_str} and {base_upper_bound_str} dollars, otherwise no."
        source_instruction = f"Please say yes only if it costs between {source_lower_bound_str} and {source_upper_bound_str} dollars, otherwise no."
        
        base_alpaca_prompt = alpaca_prompt_template % (base_instruction, base_amount_str)
        source_alpaca_prompt = alpaca_prompt_template % (source_instruction, source_amount_str)
        
        base_input_ids = tokenizer(base_alpaca_prompt, return_tensors="pt").input_ids[0]
        source_input_ids = tokenizer(source_alpaca_prompt, return_tensors="pt").input_ids[0]
        base_input_ids = base_input_ids.tolist()
        source_input_ids = source_input_ids.tolist()
        ctf_output_ids = (torch.ones(len(base_input_ids))*-100).long().tolist()
        ctf_output_ids[-1] = ctf_label
        
        all_base_input_ids += [base_input_ids]
        all_source_input_ids += [source_input_ids]
        all_ctf_output_ids += [ctf_output_ids]
        all_intervention_ids += [0]
        assert len(base_input_ids) == 82
        assert len(source_input_ids) == 82
        
    return all_base_input_ids, all_source_input_ids, all_ctf_output_ids, all_intervention_ids

def prepare_dataloader(args, tokenizer):
    prealign_batch_size = args.eval_batch_size
    logger.info(
        f"""
        Task Info:
        name = {args.task_name}
        """
    )
    raw_prealign = factual_sampler(
        tokenizer,
        args.n_eval_examples,
        game=args.task_name,
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
    
    if args.task_name == "pricing_tag_lb":
        raw_data = bound_alignment_sampler(
            tokenizer,
            args.n_training_examples+args.n_eval_examples+1000,
            [lower_bound_alignment_example_sampler]
        )
    elif args.task_name == "pricing_tag_ub":
        raw_data = bound_alignment_sampler(
            tokenizer,
            args.n_training_examples+args.n_eval_examples+1000,
            [upper_bound_alignment_example_sampler]
        )
    elif args.task_name == "pricing_tag_lub":
        raw_data = bound_alignment_sampler(
            tokenizer,
            args.n_training_examples+args.n_eval_examples+1000,
            [lower_bound_alignment_example_sampler, upper_bound_alignment_example_sampler]
        )
    elif args.task_name == "pricing_tag_mid_diff":
        raw_data = midpoint_alignment_sampler(
            tokenizer,
            args.n_training_examples+args.n_eval_examples+1000,
        )
    elif args.task_name == "pricing_tag_bracket":
        raw_data = bracket_alignment_sampler(
            tokenizer,
            args.n_training_examples+args.n_eval_examples+1000,
        )
    elif args.task_name == "pricing_tag_fixed":
        raw_data = bound_alignment_sampler(
            tokenizer,
            args.n_training_examples+args.n_eval_examples+1000,
            [lower_bound_alignment_example_sampler],
            amount=None,
            lower_bound=5.49,
            bound_width=3.00,
        )

    raw_train = (
        raw_data[0][:args.n_training_examples], 
        raw_data[1][:args.n_training_examples], 
        raw_data[2][:args.n_training_examples],
        raw_data[3][:args.n_training_examples]
    )
    raw_eval = (
        raw_data[0][args.n_training_examples:args.n_training_examples+args.n_eval_examples], 
        raw_data[1][args.n_training_examples:args.n_training_examples+args.n_eval_examples], 
        raw_data[2][args.n_training_examples:args.n_training_examples+args.n_eval_examples],
        raw_data[3][args.n_training_examples:args.n_training_examples+args.n_eval_examples]
    )
    raw_test = (
        raw_data[0][args.n_training_examples+args.n_eval_examples:], 
        raw_data[1][args.n_training_examples+args.n_eval_examples:], 
        raw_data[2][args.n_training_examples+args.n_eval_examples:],
        raw_data[3][args.n_training_examples+args.n_eval_examples:]
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
    test_dataset = Dataset.from_dict(
        {
            "input_ids": raw_test[0], 
            "source_input_ids": raw_test[1],
            "labels": raw_test[2],
            "intervention_ids": raw_test[3],
        }
    ).with_format("torch")
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.eval_batch_size,
    )
    return prealign_dataloader, train_dataloader, eval_dataloader, test_dataloader

def sample_with_region_with_triples(
    region,
    triples
):
    return random.choice(list(triples[region]))

def lower_bound_alignment_example_sampler_with_triples(
    tokenizer,
    triples,
):
    
    ctf_label_str = random.choice(["Yes", "No"])
    if ctf_label_str == "Yes":
        ctf_label = tokenizer.convert_tokens_to_ids("Yes")
        base_source_regions = [
            [1,2],
            [1,3],
            [2,2],
        ]
    elif ctf_label_str == "No":
        ctf_label = tokenizer.convert_tokens_to_ids("No")
        base_source_regions = [
            [1,1],
            [2,1],
            [2,3],
            [3,1],
            [3,2],
            [3,3]
        ]
    base_source_region = random.choice(base_source_regions)
    base_region = base_source_region[0]
    source_region = base_source_region[1]

    base_triples = sample_with_region_with_triples(base_region, triples)
    source_triples = sample_with_region_with_triples(source_region, triples)
    
    base_lower_bound_sample, base_upper_bound_sample, base_amount_sample = \
        base_triples
    source_lower_bound_sample, source_upper_bound_sample, source_amount_sample = \
        source_triples
        
    return base_lower_bound_sample, base_upper_bound_sample, \
        source_lower_bound_sample, source_upper_bound_sample, \
        base_amount_sample, source_amount_sample, ctf_label, ctf_label_str

def upper_bound_alignment_example_sampler_with_triples(
    tokenizer,
    triples,
):
    ctf_label_str = random.choice(["Yes", "No"])
    if ctf_label_str == "Yes":
        ctf_label = tokenizer.convert_tokens_to_ids("Yes")
        base_source_regions = [
            [3,2],
            [3,1],
            [2,2],
        ]
    elif ctf_label_str == "No":
        ctf_label = tokenizer.convert_tokens_to_ids("No")
        base_source_regions = [
            [1,1],
            [1,2],
            [1,3],
            [2,1],
            [2,3],
            [3,3]
        ]
    base_source_region = random.choice(base_source_regions)
    base_region = base_source_region[0]
    source_region = base_source_region[1]
    
    base_triples = sample_with_region_with_triples(base_region, triples)
    source_triples = sample_with_region_with_triples(source_region, triples)
    
    base_lower_bound_sample, base_upper_bound_sample, base_amount_sample = \
        base_triples
    source_lower_bound_sample, source_upper_bound_sample, source_amount_sample = \
        source_triples
        
    return base_lower_bound_sample, base_upper_bound_sample, \
        source_lower_bound_sample, source_upper_bound_sample, \
        base_amount_sample, source_amount_sample, ctf_label, ctf_label_str

def bound_alignment_sampler_with_triples(
    tokenizer,
    max_n_training_examples,
    bound_functors,
    triples
):
    all_base_input_ids = []
    all_source_input_ids = []
    all_ctf_output_ids = [] # this one does not have input ids, etc..
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

        base_amount_str = "%.2f dollars" % base_amount_sample
        source_amount_str = "%.2f dollars" % source_amount_sample
        base_lower_bound_str = "%.2f" % base_lower_bound_sample
        base_upper_bound_str = "%.2f" % base_upper_bound_sample
        source_lower_bound_str = "%.2f" % source_lower_bound_sample
        source_upper_bound_str = "%.2f" % source_upper_bound_sample
        
        # print(f"base: [{base_lower_bound_str}, {base_upper_bound_str}], {base_amount_str}")
        # print(f"source: [{source_lower_bound_str}, {source_upper_bound_str}], {source_amount_str}")
        # print(f"ctf label: {ctf_label_str}")
        
        base_instruction = f"Please say yes only if it costs between {base_lower_bound_str} and {base_upper_bound_str} dollars, otherwise no."
        source_instruction = f"Please say yes only if it costs between {source_lower_bound_str} and {source_upper_bound_str} dollars, otherwise no."
        
        base_alpaca_prompt = alpaca_prompt_template % (base_instruction, base_amount_str)
        source_alpaca_prompt = alpaca_prompt_template % (source_instruction, source_amount_str)
        
        base_input_ids = tokenizer(base_alpaca_prompt, return_tensors="pt").input_ids[0]
        source_input_ids = tokenizer(source_alpaca_prompt, return_tensors="pt").input_ids[0]
        base_input_ids = base_input_ids.tolist()
        source_input_ids = source_input_ids.tolist()
        ctf_output_ids = (torch.ones(len(base_input_ids))*-100).long().tolist()
        ctf_output_ids[-1] = ctf_label
        intervention_id = 0 if bound_functor == bound_functors[0] else 1
        
        all_base_input_ids += [base_input_ids]
        all_source_input_ids += [source_input_ids]
        
        all_ctf_output_ids += [ctf_output_ids]
        all_intervention_ids += [intervention_id]
        
        assert len(base_input_ids) == 82
        assert len(source_input_ids) == 82
        
    return all_base_input_ids, all_source_input_ids, all_ctf_output_ids, all_intervention_ids