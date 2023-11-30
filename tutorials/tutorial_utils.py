import itertools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from functools import partial
from typing import Dict, Optional, Sequence
from torch.nn import functional as F
import re
import evaluate
import os, random, argparse, sys, pickle, time, datasets, json
import copy, torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import Dataset 
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from collections import Counter
import networkx as nx
import ipywidgets as widgets
from ipywidgets import interact
from matplotlib.patches import Rectangle
IGNORE_INDEX = -100

"""
This is for tutorial

If the cost is between X and Y.ipynb

These dataset creation functions are copied from
https://github.com/frankaging/align-transformers/blob/cf93a1a6491dba65e1422fe20428f5972d17137e/counterfactual_datasets/price_tagging_game.py
"""

alpaca_prompt_template = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
%s

### Input:
%s

### Response:
"""


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


def chunk(iterable, chunksize):
    # if iterable is a list, we chunk with simple list indexing
    if isinstance(iterable, list):
        return [iterable[i:i+chunksize] for i in range(0, len(iterable), chunksize)]
    # otherwise if iterable is a Hf Dataset, we leverage the select() function to create mini datasets
    elif isinstance(iterable, Dataset):
        chunks = []
        for i in range(0, len(iterable), chunksize):
            if i+chunksize < len(iterable):
                chunks.append(iterable.select(list(range(i, i+chunksize))))
            else:
                chunks.append(iterable.select(list(range(i, len(iterable)))))
        return chunks
    else:
        raise Exception(f"Unrecognizable type of iterable for batchification: {type(iterable)}")
        

def reject_sample(lst, exception):
    while True:
        choice = random.choice(lst)
        if choice not in exception:
            return choice
        
        
def sample_factual_inputs(program, all_vocab, synonyms_pairs, synonyms_dict):
    
    value_map = {}
    op_map = {}
    op_value_map = {}
    
    input1, input2, input3, input4, input5 = None, None, None, None, None 
    input_id1 = program[0][2][0]
    input_id2 = program[0][2][1]
    input_id3 = program[0][2][2]
    input_id4 = program[0][0][0]
    input_id5 = program[0][0][1]
    op1 = program[0][-1][0]
    op2 = program[0][-1][1]
    op3 = program[0][1]
    op4 = program[1][-1]
    op5 = program[-1][-1]
    
    is_op1 = True if random.random() > 0.5 else False
    is_op2 = True if random.random() > 0.5 else False
    is_synonym = True if random.random() > 0.5 else False
    
    if (op1 == "==" and is_op1) or (op1 == "!=" and not is_op1):
        input1 = random.choice(all_vocab)
        input2 = input1
    elif (op1 == "!=" and is_op1) or (op1 == "==" and not is_op1):
        if is_synonym:
            input1 = random.choice(all_vocab)
            input2 = reject_sample(all_vocab, exception=[input1])
        else:
            synonyms_pair = random.choice(synonyms_pairs)
            input1 = synonyms_pair[0]
            input2 = synonyms_pair[1]

    value_map[input_id1] = input1
    value_map[input_id2] = input2
    
    if (op2 == "==" and is_op2) or (op2 == "!=" and not is_op2):
        input3 = input2
    elif (op2 == "!=" and is_op2) or (op2 == "==" and not is_op2):
        if input2 in synonyms_dict and not is_synonym:
            input3 = random.choice(synonyms_dict[input2])
        else:
            # here, we can either be the same as input1, or random sample
            if input1 != input2:
                input3 = input1 if random.random() > 0.5 else reject_sample(all_vocab, exception=[input2])
            else:
                input3 = reject_sample(all_vocab, exception=[input2])
    value_map[input_id3] = input3
    
    if is_synonym:
        synonyms_pair = random.choice(synonyms_pairs)
        input4 = synonyms_pair[0]
        input5 = synonyms_pair[1]
    else:
        input4 = random.choice(all_vocab)
        excludes = [input4]
        if input4 in synonyms_dict:
            excludes.extend(synonyms_dict[input4])
        input5 = reject_sample(all_vocab, exception=excludes)
    value_map[input_id4] = input4
    value_map[input_id5] = input5

    op_map["op1"] = op1
    op_map["op2"] = op2
    op_map["op3"] = op3
    op_map["op4"] = op4
    op_map["op5"] = op5
    
    op_value_map["op1"] = is_op1
    op_value_map["op2"] = is_op2
    op_value_map["op3"] = is_synonym
    
    first_level_values = [is_op1, is_op2, is_synonym]
    arg1_value = first_level_values[program[1][0][0]-5]
    arg2_value = first_level_values[program[1][0][1]-5]
    arg3_value = first_level_values[program[-1][0]-5]
    
    op_value_map["op4"] = (arg1_value or arg2_value) if op4 == "OR" else \
        (arg1_value and arg2_value)
    op_value_map["op5"] = (op_value_map["op4"] or arg3_value) if op5 == "OR" else \
        (op_value_map["op4"] and arg3_value)
    
    for k, v in value_map.items():
        op_value_map[k] = v # this is a more complete map.
    return program, value_map, op_map, op_value_map


def fetch_metadata(input_dir, use_token=True):
    
    synonyms_path = os.path.join(input_dir, 'token_cos_synonyms.txt' if use_token else 'synonyms.txt')
    with open(synonyms_path, 'r') as file:
        synonyms_lines = file.readlines()
    if not use_token:
        antonyms_path = os.path.join(input_dir, 'antonyms.txt')
        with open(antonyms_path, 'r') as file:
            antonyms_lines = file.readlines()

    synonyms_pairs_uni = set(
        [tuple(sorted(l.strip().split(" - "))) for l in synonyms_lines]
    )
    synonyms_pairs = set([])

    synonyms_dict = {}
    for pair in synonyms_pairs_uni:
        if pair[0] == pair[1]:
            continue
        if pair[0] not in synonyms_dict:
            synonyms_dict[pair[0]] = {pair[1]}
        else:
            synonyms_dict[pair[0]].add(pair[1])
        if pair[1] not in synonyms_dict:
            synonyms_dict[pair[1]] = {pair[0]}
        else:
            synonyms_dict[pair[1]].add(pair[0])
        synonyms_pairs.add((pair[0], pair[1]))
        synonyms_pairs.add((pair[1], pair[0]))
    synonyms_pairs = list(synonyms_pairs)
    for k, v in synonyms_dict.items():
        synonyms_dict[k] = list(v)

    all_vocab = set([])
    for pair in synonyms_pairs:
        all_vocab.add(pair[0])
        all_vocab.add(pair[1])
    if not use_token:
        for l in antonyms_lines:
            pair = l.strip().split(" - ")
            all_vocab.add(pair[0])
            all_vocab.add(pair[1])
    all_vocab = list(all_vocab)
    
    return all_vocab, synonyms_pairs, synonyms_dict


def sample_factual_input_instruction(program):
    digit_to_word_mapping = {
        0 : "first", 
        1 : "second", 
        2 : "third", 
        3 : "fourth",
        4 : "fifth"
    }
    
    first_same_or_not = "the same" if program[0][3][0] == "==" else "not the same"
    second_same_or_not = "the same" if program[0][3][1] == "==" else "not the same"
    second_part = f"A is True if "\
                  f"the {digit_to_word_mapping[program[0][2][0]]} word "\
                  f"and the {digit_to_word_mapping[program[0][2][1]]} word are {first_same_or_not}, otherwise False."
    third_part = f"B is True if "\
                  f"the {digit_to_word_mapping[program[0][2][1]]} word "\
                  f"and the {digit_to_word_mapping[program[0][2][2]]} word are {second_same_or_not}, otherwise False."
    
    first_part = f"C is True if "\
                 f"the {digit_to_word_mapping[program[0][0][0]]} word "\
                 f"and the {digit_to_word_mapping[program[0][0][1]]} word are synonyms, otherwise False."
    
    first_logic_gate = "and" if program[1][-1] == "AND" else "or"
    second_logic_gate = "and" if program[-1][-1] == "AND" else "or"
    
    first_left_var = "A" if program[1][0][0] == 5 else "B" if program[1][0][0] == 6 else "C"
    first_right_var = "A" if program[1][0][1] == 5 else "B" if program[1][0][1] == 6 else "C"
    second_var = "A" if program[-1][0] == 5 else "B" if program[-1][0] == 6 else "C"
    
    fourth_part = f"D is True if "\
                  f"{first_left_var} {first_logic_gate} {first_right_var} is True, otherwise False."
    
    fifth_part = f"The output is True if "\
                  f"{second_var} {second_logic_gate} D is True, otherwise False."
    
    # test_demo = "What is the output? "
    
    program_str = "\n".join([second_part, third_part, first_part, fourth_part, fifth_part])
    return program_str


"""
To make this more compatible with HF, maybe let's 
make this as a seq to seq task where the input is 
a sentence of words, and the output is also a single
word in this case.
"""

def prepare_factual_training_data_single_program(
    program, 
    n_sample, 
    n_in_context_demo,
    data_path=".",
    input_trigger="",
    output_trigger="=",
    program_uuid=None,
    mode="E" # ["[E]xplainations", "[N]umber"]
):
    task_instruction = sample_factual_input_instruction(program)
    instruction_template = """%s
%s"""
    all_vocab, synonyms_pairs, synonyms_dict = fetch_metadata(data_path, use_token=True)
    demo_sep="\n"
    count = 0
    examples = []
    unique_hash_set = set([])
    while len(examples) < n_sample:
        if mode == "E":
            demos = []
            for _ in range(n_in_context_demo):
                program, inputs, op_maps, value_maps = sample_factual_inputs(
                    program, all_vocab, synonyms_pairs, synonyms_dict
                )
                input_words = [inputs[i] for i in range(len(inputs))]
                input_sentence = ",".join(input_words) 
                output_word = value_maps[f'op{len(inputs)}']
                single_demo = f"{input_trigger}{input_sentence}{output_trigger}{output_word}"
                demos += [single_demo]
            program, inputs, op_maps, value_maps = sample_factual_inputs(
                program, all_vocab, synonyms_pairs, synonyms_dict
            )
            test_words = [inputs[i] for i in range(len(inputs))]
            test_sentence = ",".join(test_words) 
            test_demo = f"{input_trigger}{test_sentence}"
            demos = "\n".join(demos)
            answers = value_maps[f'op{len(inputs)}']
            question = instruction_template % (demos, test_demo)
            task_name = "word_logic_E"
        elif mode == "N":
            assert program_uuid is not None, "The mode N requires a program UUID."
            program, inputs, op_maps, value_maps = sample_factual_inputs(
                program, all_vocab, synonyms_pairs, synonyms_dict
            )
            test_words = [inputs[i] for i in range(len(inputs))]
            test_sentence = ",".join(test_words) 
            test_demo = f"{input_trigger}{test_sentence}"
            answers = value_maps[f'op{len(inputs)}']
            question = instruction_template % (program_uuid, test_demo)
            task_name = "word_logic_N"
        else:
            assert False, f"The mode {mode} is unknown."

        if f"{question} {answers}" not in unique_hash_set:
            example = {
                "question": question,
                "answers": answers,
                "program": program,
                "task_name": task_name
            }
            examples += [example]
            unique_hash_set.add(f"{question} {answers}")
    return examples


def make_icl_test_data_module(
    programs,
    num_of_programs,
    num_of_shots,
    mode,
    n_training_examples_per_program,
    tokenizer,
    data_path="./data_files/",
    select_programs=None
) -> Dict:
    clm_new_token_trigger = "="
    
    input_output_dict = {
        "question": [],
        "answers": []
    }
    
    if select_programs is None:
        program_pool = [k for k in programs.keys()]
        select_programs = random.sample(program_pool, k=num_of_programs)

    for p in select_programs:
        generated_examples = prepare_factual_training_data_single_program(
            programs[p],
            n_training_examples_per_program, 
            num_of_shots,
            mode=mode,
            program_uuid=p,
            data_path=data_path,
        )
        for e in generated_examples:
            input_output_dict["question"] += [e["question"]]
            input_output_dict["answers"] += [e["answers"]]

    raw_dataset = Dataset.from_dict(input_output_dict)
    
    def preprocess_function(examples):

        sources = examples["question"]
        targets = examples["answers"]
        # We added in a '=' to be the trigger word of answer.
        examples = [s + f"{clm_new_token_trigger}" + f"{t}{tokenizer.eos_token}" for s, t in zip(sources, targets)]

        examples_tokenized = tokenizer(
            examples,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        sources_tokenized = tokenizer(
            sources,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        labels = copy.deepcopy(examples_tokenized["input_ids"])

        for i in range(len(sources_tokenized["input_ids"])):
            source_len = len(sources_tokenized["input_ids"][i]) + 1 
            # let's not predict the trigger.
            # 1 here is a little hacky... please not follow this.
            labels_t = torch.tensor(labels[i])
            labels_t[:source_len] = IGNORE_INDEX
            labels[i] = labels_t.tolist()
        examples_tokenized["labels"] = labels

        return examples_tokenized

    test_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        desc="Running tokenizer on the test dataset",
    )
    
    return test_dataset, select_programs


def make_icl_data_module(
    programs,
    num_of_programs,
    num_of_shots,
    mode,
    n_training_examples_per_program,
    tokenizer,
    data_path="./data_files/",
    select_programs=None
) -> Dict:
    clm_new_token_trigger = "="
    
    input_output_dict = {
        "question": [],
        "answers": []
    }
    
    if select_programs is None:
        program_pool = [k for k in programs.keys()]
        select_programs = random.sample(program_pool, k=num_of_programs)

    for p in select_programs:
        generated_examples = prepare_factual_training_data_single_program(
            programs[p],
            n_training_examples_per_program, 
            num_of_shots,
            mode=mode,
            program_uuid=p,
            data_path=data_path,
        )
        for e in generated_examples:
            input_output_dict["question"] += [e["question"]]
            input_output_dict["answers"] += [e["answers"]]

    raw_dataset = Dataset.from_dict(input_output_dict)
    raw_dataset = raw_dataset.train_test_split(test_size=1000)
    test_dataset = raw_dataset["test"]
    raw_dataset = raw_dataset["train"].train_test_split(test_size=1000)
    validation_dataset = raw_dataset["test"]
    train_dataset = raw_dataset["train"]
    
    def preprocess_function(examples):

        sources = examples["question"]
        targets = examples["answers"]
        # We added in a '=' to be the trigger word of answer.
        examples = [s + f"{clm_new_token_trigger}" + f"{t}{tokenizer.eos_token}" for s, t in zip(sources, targets)]

        examples_tokenized = tokenizer(
            examples,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        sources_tokenized = tokenizer(
            sources,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        labels = copy.deepcopy(examples_tokenized["input_ids"])

        for i in range(len(sources_tokenized["input_ids"])):
            source_len = len(sources_tokenized["input_ids"][i]) + 1 
            # let's not predict the trigger.
            # 1 here is a little hacky... please not follow this.
            labels_t = torch.tensor(labels[i])
            labels_t[:source_len] = IGNORE_INDEX
            labels[i] = labels_t.tolist()
        examples_tokenized["labels"] = labels

        return examples_tokenized

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        desc="Running tokenizer on the train dataset",
    )
    validation_dataset = validation_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        desc="Running tokenizer on the validation dataset",
    )
    
    return dict(
        train_dataset=train_dataset, 
        eval_dataset=validation_dataset, 
        test_dataset=test_dataset, 
        data_collator=None
    ), select_programs

def visualize_program(
    program,
    intervene_on=None
):
    # Create an empty Graph
    G = nx.Graph()
    
    G.add_node(10, label='IN')
    G.add_node(0, label='C0')
    G.add_node(1, label='C1')
    G.add_node(2, label='C2')
    G.add_node(3, label='C3')
    G.add_node(4, label='C4')
    G.add_node(5, label=program[0][-1][0])
    G.add_node(6, label=program[0][-1][1])
    G.add_node(7, label=program[0][1])
    G.add_node(8, label=program[1][-1])
    G.add_node(9, label=program[2][-1])
    G.add_node(11, label='OUT')
    
    color_map = []
    for node in G:
        if intervene_on is None:
            color_map.append('blue')
        else:
            if node == intervene_on:
                color_map.append('red')
            else: 
                color_map.append('blue') 
    
    # Add edges
    G.add_edge(program[0][2][0], 5)
    G.add_edge(program[0][2][1], 5)
    G.add_edge(program[0][2][1], 6)
    G.add_edge(program[0][2][2], 6)

    G.add_edge(program[0][0][0], 7)
    G.add_edge(program[0][0][1], 7)

    G.add_edge(program[1][0][0], 8)
    G.add_edge(program[1][0][1], 8)

    G.add_edge(8, 9)
    G.add_edge(program[-1][0], 9)
    G.add_edge(9, 11)
    
    G.add_edge(10, 0)
    G.add_edge(10, 1)
    G.add_edge(10, 2)
    G.add_edge(10, 3)
    G.add_edge(10, 4)

    pos = {
        0: (0, 0), 1: (3, 0), 2: (6, 0), 3: (9, 0), 4: (12, 0),
        5: (1.5, 1), 6: (4.5, 1), 7: (10.5, 1),
        8: (6, 2),
        9: (6, 3),
        10: (6, -1), 11: (6, 4)
    }
    
    fig = plt.figure(1, figsize=(5, 5), dpi=100)
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=800, alpha=0.2, node_color=color_map)

    # edges
    nx.draw_networkx_edges(
        G, pos, width=2.0, alpha=0.5, edge_color='b', arrows=True,
        arrowstyle='->', arrowsize=20,
        edge_cmap=plt.cm.Blues, min_source_margin=0, min_target_margin=0
    )

    # labels
    labels = {node: data['label'] for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12)

    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.1)
    plt.axis("off")
    plt.show()
    
    
def fetch_counterfactual_value_input_only(
    base_value_maps, 
    source_value_maps, 
    program,
    intervention_on,
    fetch_on,
    all_vocab, synonyms_pairs, synonyms_dict
):
    intervention_on = int(intervention_on[-1]) # rewrite
    value_map = copy.deepcopy(base_value_maps)
    value_map[intervention_on] = source_value_maps[intervention_on]
    is_op1 = eval(f"'{value_map[program[0][2][0]]}'{program[0][-1][0]}'{value_map[program[0][2][1]]}'")
    is_op2 = eval(f"'{value_map[program[0][2][1]]}'{program[0][-1][1]}'{value_map[program[0][2][2]]}'")
    is_synonym = True if ((value_map[program[0][0][0]], value_map[program[0][0][1]]) in synonyms_pairs) or \
        ((value_map[program[0][0][1]], value_map[program[0][0][0]]) in synonyms_pairs) else False
    first_level_values = [is_op1, is_op2, is_synonym]
    arg1_value = first_level_values[program[1][0][0]-5]
    arg2_value = first_level_values[program[1][0][1]-5]
    arg3_value = first_level_values[program[-1][0]-5]
    op4 = program[1][-1]
    op5 = program[-1][-1]
    op4_value = (arg1_value or arg2_value) if op4 == "OR" else \
        (arg1_value and arg2_value)
    op5_value = (op4_value or arg3_value) if op5 == "OR" else \
        (op4_value and arg3_value)
    
    return {'op1': is_op1, 'op2': is_op2, 'op3': is_synonym, 'op4': op4_value, 'op5': op5_value}[fetch_on]

def fetch_counterfactual_value(
    base_op_value_maps, source_op_value_maps, program, 
    intervention_on, fetch_on,
    all_vocab, synonyms_pairs, synonyms_dict
):
    if intervention_on in {"C0", "C1", "C2", "C3", "C4"}:
        return fetch_counterfactual_value_input_only(
            base_op_value_maps, source_op_value_maps, program, 
            intervention_on, fetch_on,
            all_vocab, synonyms_pairs, synonyms_dict
        )
    
    intervened_value_maps = copy.deepcopy(base_op_value_maps)
    intervened_value_maps[intervention_on] = source_op_value_maps[intervention_on]
    
    op1 = program[0][-1][0]
    op2 = program[0][-1][1]
    op3 = program[0][1]
    op4 = program[1][-1]
    op5 = program[-1][-1]
    first_level_values = [
        intervened_value_maps["op1"], 
        intervened_value_maps["op2"], 
        intervened_value_maps["op3"]
    ]
    arg1_value = first_level_values[program[1][0][0]-5]
    arg2_value = first_level_values[program[1][0][1]-5]
    arg3_value = first_level_values[program[-1][0]-5]
    
    if intervention_on in ["op1", "op2", "op3"]:
        intervened_value_maps["op4"] = (arg1_value or arg2_value) if op4 == "OR" else \
            (arg1_value and arg2_value)
        intervened_value_maps["op5"] = (intervened_value_maps["op4"] or arg3_value) if op5 == "OR" else \
            (intervened_value_maps["op4"] and arg3_value)
    elif intervention_on == "op4":
        intervened_value_maps["op5"] = (intervened_value_maps["op4"] or arg3_value) if op5 == "OR" else \
            (intervened_value_maps["op4"] and arg3_value)
    elif intervention_on == "op5":
        pass
    return intervened_value_maps[fetch_on]
    

def sample_demos(
    program, all_vocab, synonyms_pairs, 
    synonyms_dict, n_in_context_demo,
    input_trigger="",
    output_trigger="=",
):
    demos = []
    for _ in range(n_in_context_demo):
        _, inputs, _, value_maps = sample_factual_inputs(
            program, all_vocab, synonyms_pairs, synonyms_dict
        )
        input_words = [inputs[i] for i in range(len(inputs))]
        input_sentence = ",".join(input_words) 
        output_word = value_maps[f'op{len(inputs)}']
        single_demo = f"{input_trigger}{input_sentence}{output_trigger}{output_word}"
        demos += [single_demo]
    return "\n".join(demos)


def prepare_counterfactual_alignment_data(
    program, 
    n_sample, 
    n_in_context_demo,
    aligning_causal_variable,
    data_path=".",
    input_trigger="",
    output_trigger="=",
    program_uuid=None,
    mode="E" # ["[E]xplainations", "[N]umber"]
):
    task_instruction = sample_factual_input_instruction(program)
    instruction_template = """%s
%s"""
    all_vocab, synonyms_pairs, synonyms_dict = fetch_metadata(data_path, use_token=True)
    demo_sep="\n"
    count = 0
    examples = []
    unique_hash_set = set([])
    while len(examples) < n_sample:
        
        _, inputs, _, value_maps = sample_factual_inputs(
            program, all_vocab, synonyms_pairs, synonyms_dict
        )
        _, source_inputs, _, source_value_maps = sample_factual_inputs(
            program, all_vocab, synonyms_pairs, synonyms_dict
        )
        base_answers = value_maps["op5"]
        source_answers = source_value_maps["op5"]
        
        answers = fetch_counterfactual_value(
            value_maps, source_value_maps, program, 
            aligning_causal_variable, "op5",
            all_vocab, synonyms_pairs, synonyms_dict
        )

        if len(examples) < n_sample//2:
            if base_answers == answers:
                continue
        else:
            if base_answers != answers:
                continue        
        
        # construct inputs based on the mode
        if mode == "E":
            base_demos = sample_demos(
                program, all_vocab, synonyms_pairs, 
                synonyms_dict, n_in_context_demo,
            )
            source_demos = sample_demos(
                program, all_vocab, synonyms_pairs, 
                synonyms_dict, n_in_context_demo,
            )
            
            base_test_sentence = ",".join([inputs[i] for i in range(len(inputs))]) 
            base_test_demo = f"{input_trigger}{base_test_sentence}"
            base_question = instruction_template % (base_demos, base_test_demo)
            
            source_test_sentence = ",".join([source_inputs[i] for i in range(len(source_inputs))]) 
            source_test_demo = f"{input_trigger}{source_test_sentence}"
            source_question = instruction_template % (source_demos, source_test_demo)
            
            task_name = "word_logic_E"
        elif mode == "N":
            assert program_uuid is not None, "The mode N requires a program UUID."
            base_test_sentence = ",".join([inputs[i] for i in range(len(inputs))]) 
            base_test_demo = f"{input_trigger}{base_test_sentence}"
            base_question = instruction_template % (program_uuid, base_test_demo)
            
            source_test_sentence = ",".join([source_inputs[i] for i in range(len(source_inputs))]) 
            source_test_demo = f"{input_trigger}{source_test_sentence}"
            source_question = instruction_template % (program_uuid, source_test_demo)
            
            task_name = "word_logic_N"
        else:
            assert False, f"The mode {mode} is unknown."
        
        
        if f"{base_question} {source_question}" not in unique_hash_set:
            example = {
                "question": base_question,
                "source_question": source_question,
                "answers": answers,
                "base_answers": base_answers,
                "source_answers": source_answers
            }
            examples += [example]
            unique_hash_set.add(f"{base_question} {source_question}")
    
    input_output_dict = {
        "question": [],
        "source_question": [],
        "answers": [],
        "base_answers": [],
        "source_answers": []
    }
    for e in examples:
        input_output_dict["question"] += [e["question"]]
        input_output_dict["source_question"] += [e["source_question"]]
        input_output_dict["answers"] += [e["answers"]]
        input_output_dict["base_answers"] += [e["base_answers"]]
        input_output_dict["source_answers"] += [e["source_answers"]]
        
    return input_output_dict


def prepare_mismatch_counterfactual_alignment_data(
    program, 
    mismatch_program, 
    n_sample, 
    n_in_context_demo,
    aligning_causal_variable,
    data_path=".",
    input_trigger="",
    output_trigger="=",
    program_uuid=None,
    mode="E" # ["[E]xplainations", "[N]umber"]
):
    task_instruction = sample_factual_input_instruction(program)
    instruction_template = """%s
%s"""
    all_vocab, synonyms_pairs, synonyms_dict = fetch_metadata(data_path, use_token=True)
    demo_sep="\n"
    count = 0
    examples = []
    unique_hash_set = set([])
    while len(examples) < n_sample:
        
        # the counterfactual lable is from the mismatched program
        _, inputs, _, value_maps = sample_factual_inputs(
            mismatch_program, all_vocab, synonyms_pairs, synonyms_dict
        )
        _, source_inputs, _, source_value_maps = sample_factual_inputs(
            mismatch_program, all_vocab, synonyms_pairs, synonyms_dict
        )
        base_answers = value_maps["op5"]
        source_answers = source_value_maps["op5"]
        answers = fetch_counterfactual_value(
            value_maps, source_value_maps, mismatch_program, 
            aligning_causal_variable, "op5",
            all_vocab, synonyms_pairs, synonyms_dict
        )

        if len(examples) < n_sample//2:
            if base_answers == answers:
                continue
        else:
            if base_answers != answers:
                continue        
        
        # construct inputs based on the mode
        if mode == "E":
            base_demos = sample_demos(
                program, all_vocab, synonyms_pairs, 
                synonyms_dict, n_in_context_demo,
            )
            source_demos = sample_demos(
                program, all_vocab, synonyms_pairs, 
                synonyms_dict, n_in_context_demo,
            )
            
            base_test_sentence = ",".join([inputs[i] for i in range(len(inputs))]) 
            base_test_demo = f"{input_trigger}{base_test_sentence}"
            base_question = instruction_template % (base_demos, base_test_demo)
            
            source_test_sentence = ",".join([source_inputs[i] for i in range(len(source_inputs))]) 
            source_test_demo = f"{input_trigger}{source_test_sentence}"
            source_question = instruction_template % (source_demos, source_test_demo)
            
            task_name = "word_logic_E"
        elif mode == "N":
            assert program_uuid is not None, "The mode N requires a program UUID."
            base_test_sentence = ",".join([inputs[i] for i in range(len(inputs))]) 
            base_test_demo = f"{input_trigger}{base_test_sentence}"
            base_question = instruction_template % (program_uuid, base_test_demo)
            
            source_test_sentence = ",".join([source_inputs[i] for i in range(len(source_inputs))]) 
            source_test_demo = f"{input_trigger}{source_test_sentence}"
            source_question = instruction_template % (program_uuid, source_test_demo)
            
            task_name = "word_logic_N"
        else:
            assert False, f"The mode {mode} is unknown."
        
        
        if f"{base_question} {source_question}" not in unique_hash_set:
            example = {
                "question": base_question,
                "source_question": source_question,
                "answers": answers,
                "base_answers": base_answers,
                "source_answers": source_answers
            }
            examples += [example]
            unique_hash_set.add(f"{base_question} {source_question}")
    
    input_output_dict = {
        "question": [],
        "source_question": [],
        "answers": [],
        "base_answers": [],
        "source_answers": []
    }
    for e in examples:
        input_output_dict["question"] += [e["question"]]
        input_output_dict["source_question"] += [e["source_question"]]
        input_output_dict["answers"] += [e["answers"]]
        input_output_dict["base_answers"] += [e["base_answers"]]
        input_output_dict["source_answers"] += [e["source_answers"]]
        
    return input_output_dict


def make_mismatch_supervised_counterfactual_data_module(
    program, 
    mismatch_program, 
    n_sample, 
    n_in_context_demo,
    aligning_causal_variable,
    tokenizer,
    data_path=".",
    input_trigger="",
    output_trigger="=",
    program_uuid=None,
    mode="E", # ["[E]xplainations", "[N]umber"]
    n_test_sample=5000
):
    clm_new_token_trigger = "="
    
    train_cdataset = Dataset.from_dict(
        prepare_mismatch_counterfactual_alignment_data(
            program, 
            mismatch_program, 
            n_sample, 
            n_in_context_demo,
            aligning_causal_variable,
            data_path,
            input_trigger,
            output_trigger,
            program_uuid,
            mode
        )
    )
    
    validation_cdataset = Dataset.from_dict(
        prepare_mismatch_counterfactual_alignment_data(
            program, 
            mismatch_program, 
            n_test_sample, 
            n_in_context_demo,
            aligning_causal_variable,
            data_path,
            input_trigger,
            output_trigger,
            program_uuid,
            mode
        )
    )
    test_cdataset = Dataset.from_dict(
        prepare_mismatch_counterfactual_alignment_data(
            program, 
            mismatch_program, 
            n_test_sample, 
            n_in_context_demo,
            aligning_causal_variable,
            data_path,
            input_trigger,
            output_trigger,
            program_uuid,
            mode
        )
    )
    
    def counterfactual_preprocess_function(
        examples,
    ):
        base_examples = [
            s + f"{clm_new_token_trigger}" + f"{t}" for s, t in zip(examples["question"], examples["base_answers"])]
        source_examples = [
            s + f"{clm_new_token_trigger}" + f"{t}" for s, t in zip(examples["source_question"], examples["source_answers"])]
        counterfactual_examples = [
            s + f"{clm_new_token_trigger}" + f"{t}" for s, t in zip(examples["question"], examples["answers"])]
        
        base_examples_tokenized = tokenizer(
            base_examples,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        source_examples_tokenized = tokenizer(
            source_examples,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        counterfactual_examples_tokenized = tokenizer(
            counterfactual_examples,
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) # we only need the label      
        counterfactual_labels = copy.deepcopy(
            counterfactual_examples_tokenized["input_ids"])

        for i in range(len(counterfactual_labels)):
            labels_t = torch.tensor(counterfactual_labels[i])
            labels_t[:-1] = IGNORE_INDEX # the last one is the counterfactual label id
            counterfactual_labels[i] = labels_t.tolist()

        examples["input_ids"] = base_examples_tokenized["input_ids"]
        examples["attention_mask"] = base_examples_tokenized["attention_mask"]
        examples["source_input_ids"] = source_examples_tokenized["input_ids"]
        examples["source_attention_mask"] = source_examples_tokenized["attention_mask"]
        examples["labels"] = counterfactual_labels

        return examples

    remove_columns=[]
    train_cdataset = train_cdataset.map(
        counterfactual_preprocess_function,
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=remove_columns,
        desc="Running tokenizer on the train dataset",
    )
    validation_cdataset = validation_cdataset.map(
        counterfactual_preprocess_function,
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=remove_columns,
        desc="Running tokenizer on the validation dataset",
    )
    test_cdataset = test_cdataset.map(
        counterfactual_preprocess_function,
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=remove_columns,
        desc="Running tokenizer on the test dataset",
    )
    
    return dict(
        train_dataset=train_cdataset, 
        eval_dataset=validation_cdataset, 
        test_dataset=test_cdataset, 
        data_collator=None
    ), (program, mismatch_program)


def make_supervised_counterfactual_data_module(
    program, 
    n_sample, 
    n_in_context_demo,
    aligning_causal_variable,
    tokenizer,
    data_path=".",
    input_trigger="",
    output_trigger="=",
    program_uuid=None,
    mode="E", # ["[E]xplainations", "[N]umber"]
    n_test_sample=5000
):
    clm_new_token_trigger = "="
    
    train_cdataset = Dataset.from_dict(
        prepare_counterfactual_alignment_data(
            program, 
            n_sample, 
            n_in_context_demo,
            aligning_causal_variable,
            data_path,
            input_trigger,
            output_trigger,
            program_uuid,
            mode
        )
    )
    
    validation_cdataset = Dataset.from_dict(
        prepare_counterfactual_alignment_data(
            program, 
            n_test_sample, 
            n_in_context_demo,
            aligning_causal_variable,
            data_path,
            input_trigger,
            output_trigger,
            program_uuid,
            mode
        )
    )
    test_cdataset = Dataset.from_dict(
        prepare_counterfactual_alignment_data(
            program, 
            n_test_sample, 
            n_in_context_demo,
            aligning_causal_variable,
            data_path,
            input_trigger,
            output_trigger,
            program_uuid,
            mode
        )
    )
    
    def counterfactual_preprocess_function(
        examples,
    ):
        base_examples = [
            s + f"{clm_new_token_trigger}" + f"{t}" for s, t in zip(examples["question"], examples["base_answers"])]
        source_examples = [
            s + f"{clm_new_token_trigger}" + f"{t}" for s, t in zip(examples["source_question"], examples["source_answers"])]
        counterfactual_examples = [
            s + f"{clm_new_token_trigger}" + f"{t}" for s, t in zip(examples["question"], examples["answers"])]
        
        base_examples_tokenized = tokenizer(
            base_examples,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        source_examples_tokenized = tokenizer(
            source_examples,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        counterfactual_examples_tokenized = tokenizer(
            counterfactual_examples,
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) # we only need the label      
        counterfactual_labels = copy.deepcopy(
            counterfactual_examples_tokenized["input_ids"])

        for i in range(len(counterfactual_labels)):
            labels_t = torch.tensor(counterfactual_labels[i])
            labels_t[:-1] = IGNORE_INDEX # the last one is the counterfactual label id
            counterfactual_labels[i] = labels_t.tolist()

        examples["input_ids"] = base_examples_tokenized["input_ids"]
        examples["attention_mask"] = base_examples_tokenized["attention_mask"]
        examples["source_input_ids"] = source_examples_tokenized["input_ids"]
        examples["source_attention_mask"] = source_examples_tokenized["attention_mask"]
        examples["labels"] = counterfactual_labels

        return examples

    remove_columns=[]
    train_cdataset = train_cdataset.map(
        counterfactual_preprocess_function,
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=remove_columns,
        desc="Running tokenizer on the train dataset",
    )
    validation_cdataset = validation_cdataset.map(
        counterfactual_preprocess_function,
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=remove_columns,
        desc="Running tokenizer on the validation dataset",
    )
    test_cdataset = test_cdataset.map(
        counterfactual_preprocess_function,
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=remove_columns,
        desc="Running tokenizer on the test dataset",
    )
    
    return dict(
        train_dataset=train_cdataset, 
        eval_dataset=validation_cdataset, 
        test_dataset=test_cdataset, 
        data_collator=None
    ), program


def prepare_serialized_counterfactual_alignment_data(
    program, 
    n_sample, 
    n_in_context_demo,
    aligning_causal_variable1,
    aligning_causal_variable2,
    data_path=".",
    input_trigger="",
    output_trigger="=",
    program_uuid=None,
    mode="E" # ["[E]xplainations", "[N]umber"]
):
    task_instruction = sample_factual_input_instruction(program)
    instruction_template = """%s
%s"""
    all_vocab, synonyms_pairs, synonyms_dict = fetch_metadata(data_path, use_token=True)
    demo_sep="\n"
    count = 0
    examples = []
    unique_hash_set = set([])
    while len(examples) < n_sample:
        
        _, inputs, _, value_maps = sample_factual_inputs(
            program, all_vocab, synonyms_pairs, synonyms_dict
        )
        _, source0_inputs, _, source0_value_maps = sample_factual_inputs(
            program, all_vocab, synonyms_pairs, synonyms_dict
        )
        _, source1_inputs, _, source1_value_maps = sample_factual_inputs(
            program, all_vocab, synonyms_pairs, synonyms_dict
        )
        
        base_answers = value_maps["op5"]
        source0_answers = source0_value_maps["op5"]
        source1_answers = source1_value_maps["op5"]
        
        # source1->source2
        values_causal_variable2 = fetch_counterfactual_value(
            source1_value_maps, source0_value_maps, program, 
            aligning_causal_variable1, aligning_causal_variable2,
            all_vocab, synonyms_pairs, synonyms_dict
        )
        source1_value_maps[aligning_causal_variable2] = values_causal_variable2
        # source2->base
        answers = fetch_counterfactual_value(
            value_maps, source1_value_maps, program, 
            aligning_causal_variable2, "op5",
            all_vocab, synonyms_pairs, synonyms_dict
        )

        if len(examples) < n_sample//2:
            if base_answers == answers:
                continue
        else:
            if base_answers != answers:
                continue        
        
        # construct inputs based on the mode
        if mode == "E":
            base_instruction_prompt = sample_demos(
                program, all_vocab, synonyms_pairs, 
                synonyms_dict, n_in_context_demo,
            )
            source0_instruction_prompt = sample_demos(
                program, all_vocab, synonyms_pairs, 
                synonyms_dict, n_in_context_demo,
            )
            source1_instruction_prompt = sample_demos(
                program, all_vocab, synonyms_pairs, 
                synonyms_dict, n_in_context_demo,
            )
            task_name = "word_logic_E"
        elif mode == "N":
            assert program_uuid is not None, "The mode N requires a program UUID."
            base_instruction_prompt = program_uuid
            source0_instruction_prompt = program_uuid
            source1_instruction_prompt = program_uuid
            task_name = "word_logic_N"
        else:
            assert False, f"The mode {mode} is unknown."
        
        base_test_sentence = ",".join([inputs[i] for i in range(len(inputs))]) 
        base_test_demo = f"{input_trigger}{base_test_sentence}"
        base_question = instruction_template % (base_instruction_prompt, base_test_demo)

        source0_test_sentence = ",".join([source0_inputs[i] for i in range(len(source0_inputs))]) 
        source0_test_demo = f"{input_trigger}{source0_test_sentence}"
        source0_question = instruction_template % (source0_instruction_prompt, source0_test_demo)

        source1_test_sentence = ",".join([source1_inputs[i] for i in range(len(source1_inputs))]) 
        source1_test_demo = f"{input_trigger}{source1_test_sentence}"
        source1_question = instruction_template % (source1_instruction_prompt, source1_test_demo)
        
        if f"{base_question} {source0_question} {source1_question}" not in unique_hash_set:
            example = {
                "question": base_question,
                "source0_question": source0_question,
                "source1_question": source1_question,
                "answers": answers,
                "base_answers": base_answers,
                "source0_answers": source0_answers,
                "source1_answers": source1_answers
            }
            examples += [example]
            unique_hash_set.add(f"{base_question} {source0_question} {source1_question}")
    
    input_output_dict = {
        "question": [],
        "source0_question": [],
        "source1_question": [],
        "answers": [],
        "base_answers": [],
        "source0_answers": [],
        "source1_answers": []
    }
    for e in examples:
        input_output_dict["question"] += [e["question"]]
        input_output_dict["source0_question"] += [e["source0_question"]]
        input_output_dict["source1_question"] += [e["source1_question"]]
        input_output_dict["answers"] += [e["answers"]]
        input_output_dict["base_answers"] += [e["base_answers"]]
        input_output_dict["source0_answers"] += [e["source0_answers"]]
        input_output_dict["source1_answers"] += [e["source1_answers"]]
        
    return input_output_dict


def make_supervised_serialized_counterfactual_data_module(
    program, 
    n_sample, 
    n_in_context_demo,
    aligning_causal_variable1,
    aligning_causal_variable2,
    tokenizer,
    data_path=".",
    input_trigger="",
    output_trigger="=",
    program_uuid=None,
    mode="E", # ["[E]xplainations", "[N]umber"]
    n_test_sample=5000
):
    clm_new_token_trigger = "="
    
    train_cdataset = Dataset.from_dict(
        prepare_serialized_counterfactual_alignment_data(
            program, 
            n_sample, 
            n_in_context_demo,
            aligning_causal_variable1,
            aligning_causal_variable2,
            data_path,
            input_trigger,
            output_trigger,
            program_uuid,
            mode
        )
    )
    
    validation_cdataset = Dataset.from_dict(
        prepare_serialized_counterfactual_alignment_data(
            program, 
            n_test_sample, 
            n_in_context_demo,
            aligning_causal_variable1,
            aligning_causal_variable2,
            data_path,
            input_trigger,
            output_trigger,
            program_uuid,
            mode
        )
    )
    test_cdataset = Dataset.from_dict(
        prepare_serialized_counterfactual_alignment_data(
            program, 
            n_test_sample, 
            n_in_context_demo,
            aligning_causal_variable1,
            aligning_causal_variable2,
            data_path,
            input_trigger,
            output_trigger,
            program_uuid,
            mode
        )
    )
    
    def counterfactual_preprocess_function(
        examples,
    ):
        base_examples = [
            s + f"{clm_new_token_trigger}" + f"{t}" for s, t in zip(examples["question"], examples["base_answers"])]
        source0_examples = [
            s + f"{clm_new_token_trigger}" + f"{t}" for s, t in zip(examples["source0_question"], examples["source0_answers"])]
        source1_examples = [
            s + f"{clm_new_token_trigger}" + f"{t}" for s, t in zip(examples["source1_question"], examples["source1_answers"])]
        counterfactual_examples = [
            s + f"{clm_new_token_trigger}" + f"{t}" for s, t in zip(examples["question"], examples["answers"])]
        
        base_examples_tokenized = tokenizer(
            base_examples,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        source0_examples_tokenized = tokenizer(
            source0_examples,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        source1_examples_tokenized = tokenizer(
            source1_examples,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        counterfactual_examples_tokenized = tokenizer(
            counterfactual_examples,
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) # we only need the label      
        counterfactual_labels = copy.deepcopy(
            counterfactual_examples_tokenized["input_ids"])

        for i in range(len(counterfactual_labels)):
            labels_t = torch.tensor(counterfactual_labels[i])
            labels_t[:-1] = IGNORE_INDEX # the last one is the counterfactual label id
            counterfactual_labels[i] = labels_t.tolist()

        examples["input_ids"] = base_examples_tokenized["input_ids"]
        examples["attention_mask"] = base_examples_tokenized["attention_mask"]
        examples["source0_input_ids"] = source0_examples_tokenized["input_ids"]
        examples["source0_attention_mask"] = source0_examples_tokenized["attention_mask"]
        examples["source1_input_ids"] = source1_examples_tokenized["input_ids"]
        examples["source1_attention_mask"] = source1_examples_tokenized["attention_mask"]
        examples["labels"] = counterfactual_labels

        return examples

    remove_columns=[]
    train_cdataset = train_cdataset.map(
        counterfactual_preprocess_function,
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=remove_columns,
    )
    validation_cdataset = validation_cdataset.map(
        counterfactual_preprocess_function,
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=remove_columns,
    )
    test_cdataset = test_cdataset.map(
        counterfactual_preprocess_function,
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=remove_columns,
    )
    
    return dict(
        train_dataset=train_cdataset, 
        eval_dataset=validation_cdataset, 
        test_dataset=test_cdataset, 
        data_collator=None
    ), program