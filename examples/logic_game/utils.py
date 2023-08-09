import sys
sys.path.append("../..")

import itertools
import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import Counter
import numpy as np
import seaborn as sns
from tqdm import tqdm
import json
import os
from functools import partial
from datasets import Dataset, load_dataset
import copy
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer, 
    GPT2Config, 
    GPT2LMHeadModel, 
    DataCollatorForSeq2Seq,
    Trainer, 
    TrainingArguments,
    set_seed,
    EvalPrediction,
    get_linear_schedule_with_warmup,
    logging
)

from torch.nn import functional as F
import re

from datasets import Dataset
import evaluate
import copy, torch
from tqdm import tqdm
from models.modelings_alignable import AutoAlignableModel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from trainer import Aligner, CACHE_DIR
from collections import Counter
import pandas as pd

import networkx as nx
import ipywidgets as widgets
from ipywidgets import interact
from matplotlib.patches import Rectangle

from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger("transformers")

IGNORE_INDEX = -100
SEED = 42

def plot_transformer(
    max_length=10, num_layers=12, highlights=None, xtick_labels=None,
    vmin=0.80, vmax=1.00
):
    data = np.zeros((num_layers, max_length))

    fig, ax = plt.subplots(figsize=(5, 8))

    # Add highlights
    if highlights is not None:
        for highlight in highlights:
            data[highlight[0], highlight[1]] = highlight[2]

    # Plot the heatmap
    sns.heatmap(
        data, annot=True, linewidths=1, ax=ax, 
        cmap="cividis", cbar=False, 
        vmin=vmin, vmax=vmax # this is rather abitrary to help visualizing!
    )
    
    # Customize the plot
    ax.set_xlabel('Token')
    ax.set_ylabel('Layer')
    ax.set_title('Transformer Block View (residual+mlp)')
    
    # Set aspect ratio
    ax.set_aspect('0.5')
    
    # Set x-tick labels if provided
    if xtick_labels is not None:
        ax.set_xticklabels(xtick_labels)
    
    plt.gca().invert_yaxis()

    plt.show()

def sigmoid_boundary_sigmoid(_input, boundary_x, boundary_y, temperature):
    return torch.sigmoid((_input - boundary_x) / temperature) * \
        torch.sigmoid((boundary_y - _input) / temperature)

def plot_attention_graph(max_length=10, num_layers=12, highlight_edges=None):
    # Initialize an empty directed graph
    G = nx.DiGraph()

    # Add nodes for each token at each layer
    for layer in range(num_layers):
        for token in range(max_length):
            G.add_node((layer, token))

    # Add edges to represent causal attention
    for layer in range(1, num_layers):  # Start at 1 because layer 0 has no previous layer
        for token in range(max_length):
            for prev_token in range(token + 1):  # +1 because range is exclusive at the upper end
                G.add_edge((layer - 1, prev_token), (layer, token))

    # Create a figure and a 1x1 subplot grid
    fig, ax = plt.subplots(figsize=(6, 6))

    # Remove grid and set background color to white
    ax.grid(False)
    ax.set_facecolor('white')
    
    # Position nodes to line up with heatmap, with the first layer at the top and the last layer at the bottom
    pos = {(layer, token): (token, layer) for layer in range(num_layers) for token in range(max_length)}

    # Draw the nodes on the subplot
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=100)

    # Draw the edges with specified opacity
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.2)

    # Draw the highlighted edges in red
    if highlight_edges is not None:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=highlight_edges, edge_color='red')

    # Draw the node labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6)

    # Show the plot
    plt.show()

def extract_output(pred, trigger=''):
    if not trigger:
        return pred
    # for causallm only, use special trigger to detect new tokens. See model_args.clm_new_token_trigger
    # if cannot find trigger --> generation is too long; default to empty generation
    start = pred.find(trigger)
    if start < 0:
        return ''
    output = pred[start+len(trigger):].lstrip() # left strip any whitespaces
    return output


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
    
def generate_programs():
    possible_programs = []
    for i in range(5):
        for j in range(5):
            if i != j:
                remaining_elements = {0,1,2,3,4} - {i,j}
                remaining_elements = \
                    list(itertools.permutations(
                    list(remaining_elements)))
                for k in remaining_elements:
                    for sign in [("==", "=="), ("==", "!="), ("!=", "=="), ("!=", "!=")]:
                        for lv2_combine in [(5, 6), (5, 7), (6, 7)]:
                            for lv2_sign in ["OR", "AND"]:
                                for lv3_sign in ["OR", "AND"]:
                                    lv3_element = list({5,6,7} - set(lv2_combine))[0]
                                    for entail_sign in ["s"]:
                                        possible_programs += [
                                            (((i, j), entail_sign, tuple(k), sign), (lv2_combine, lv2_sign), (lv3_element, lv3_sign))
                                        ]
    return possible_programs

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

def eval_program(program, value_map, synonyms_pairs, synonyms_dict):
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
    return op5_value

def sort_by_values_len(dict):
    dict_len= {key: len(value) for key, value in dict.items()}
    import operator
    sorted_key_list = sorted(dict_len.items(), key=operator.itemgetter(1), reverse=True)
    sorted_dict = [{item[0]: dict[item [0]]} for item in sorted_key_list]
    return sorted_dict

def weakly_correlated_variables(correlation_matrix, threshold):
    n = len(correlation_matrix)
    weak_set = set()

    for i in range(n):
        if all(abs(correlation_matrix[i][j]) < threshold for j in weak_set):
            weak_set.add(i)

    return weak_set

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


@dataclass
class DataCollatorForAlignmentDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer, label_pad_token_id, padding):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
        self.padding = padding

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_dict = self.tokenizer.pad(
            {'input_ids': [instance['input_ids'] for instance in instances]}
        )
        source_input_dict = self.tokenizer.pad(
            {'input_ids': [instance['source_input_ids'] for instance in instances]}
        )
        intervention_ids = [instance['intervention_ids'] for instance in instances]
        # note that we need to offset the range if it is left padding!
        """
        [pad, pad, pad, t1, t2, t3, t4, t5]
        
        v.s.
        
        [t1, t2, t3, t4, t5, pad, pad, pad]
        
        the original token range is for the send type of padding after all!
        """
        if self.tokenizer.padding_side == "left":
            seq_len = torch.tensor(
                input_dict["attention_mask"], dtype=torch.int64
            ).sum(dim=-1)
            source_seq_len = torch.tensor(
                source_input_dict["attention_mask"], dtype=torch.int64
            ).sum(dim=-1)
            max_seq_len = len(input_dict["attention_mask"][0])
            max_source_seq_len = len(source_input_dict["attention_mask"][0])
            token_range = [
                [
                    instances[i]['token_range'][0]+(max_seq_len-seq_len[i]), 
                    instances[i]['token_range'][1]+(max_seq_len-seq_len[i])
                ] 
                for i in range(len(instances))
            ]
            source_token_range = [
                [
                    instances[i]['source_token_range'][0]+(max_source_seq_len-source_seq_len[i]), 
                    instances[i]['source_token_range'][1]+(max_source_seq_len-source_seq_len[i])
                ] 
                for i in range(len(instances))
            ]
        else:
            token_range = [
                instance['token_range']
                for instance in instances
            ]
            source_token_range = [
                instance['source_token_range']
                for instance in instances
            ]
        
        labels = [instance['labels'] for instance in instances]
        sequence_length = torch.tensor(input_dict["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            labels = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
        else:
            labels = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]
        batch = dict(
            input_ids=input_dict["input_ids"],
            attention_mask=input_dict["attention_mask"],
            token_range=token_range,
            source_input_ids=source_input_dict["input_ids"],
            source_attention_mask=source_input_dict["attention_mask"],
            source_token_range=source_token_range,
            intervention_ids=intervention_ids,
            labels=labels,
        )        
        return {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
    
def make_supervised_data_module(
    program,
    n_training_examples,
    tokenizer
) -> Dict:
    clm_new_token_trigger = "="
    
    all_vocab, synonyms_pairs, synonyms_dict = fetch_metadata(".", use_token=True)
    input_output_dict = {
        "question": [],
        "answers": []
    }
    while len(input_output_dict["question"]) < n_training_examples:
        _, inputs, _, value_maps = sample_factual_inputs(
            program[1], all_vocab, synonyms_pairs, synonyms_dict
        )
        input_words = [inputs[i] for i in range(len(inputs))]
        input_sentence = ",".join(input_words) 
        answers = value_maps[f'op{len(inputs)}']
        if input_sentence not in input_output_dict["question"]:
            input_output_dict["question"] += [input_sentence]
            input_output_dict["answers"] += [answers]
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
    )


def prepare_counterfactual_alignment_data_simple(
    program,
    n_sample,
    aligning_causal_variable,
    all_vocab, synonyms_pairs, synonyms_dict
):
    aligning_causal_variable_map = {
        "op1": 0,
        "op2": 0,
        "op3": 0,
        "op4": 0,
        "op5": 0,
        "C0": 0,
        "C1": 0,
        "C2": 0,
        "C3": 0,
        "C4": 0
    }
    ##################################
    #
    # Try to implement this by yourself!
    # You can know more about alignment.
    #
    ##################################
    input_output_dict = {
        "question": [],
        "source_question": [],
        "intervention_ids": [],
        "answers": [],
        "base_answers": [],
        "source_answers": []
    }
    aligning_causal_variable_map = {
        "op1": 0,
        "op2": 0,
        "op3": 0,
        "op4": 0,
        "op5": 0,
        "C0": 0,
        "C1": 0,
        "C2": 0,
        "C3": 0,
        "C4": 0
    }
    ##################################
    #
    # Try to implement this by yourself!
    # You can know more about alignment.
    #
    ##################################
    input_output_dict = {
        "question": [],
        "source_question": [],
        "intervention_ids": [],
        "answers": [],
        "base_answers": [],
        "source_answers": []
    }
    while len(input_output_dict["question"]) < n_sample:
        
        _, inputs, _, value_maps = sample_factual_inputs(
            program, all_vocab, synonyms_pairs, synonyms_dict
        )

        if aligning_causal_variable in {"C0", "C1", "C2", "C3", "C4"}:
            # we use a more systematic intervention sampling strategy.
            strategy = random.choice(["change_C", "softkeep_C", "random_C"])
            aligning_idx = int(aligning_causal_variable[-1])
            associate_variable = get_associate_variables(program, aligning_causal_variable)
            associate_idx = int(associate_variable[-1])
            
            if is_not_identity_check(program, aligning_causal_variable):
                if value_maps["op4"] == True:
                    continue
                source_inputs = copy.deepcopy(inputs)
                source_value_maps = copy.deepcopy(value_maps)
                if random.random() < 0.5:
                    # semantic soft retrieve.
                    new_val = random.choice(synonyms_dict[inputs[associate_idx]])
                    source_inputs[aligning_idx] = new_val
                    source_value_maps[aligning_idx] = new_val
                else:
                    ori_value = source_inputs[aligning_idx]
                    new_value = reject_sample(all_vocab, exception=[ori_value])
                    source_inputs[aligning_idx] = new_value
                    source_value_maps[aligning_idx] = new_value
            else:
                if value_maps["op3"] == True:
                    continue
                # 1) we keep everything else the same but change C_{target}
                if strategy == "change_C":
                    source_inputs = copy.deepcopy(inputs)
                    source_value_maps = copy.deepcopy(value_maps)
                    if random.random() < 0.5:
                        # likely != True -> True
                        # likely == True -> False
                        # likely != False -> True
                        # likely == False -> False
                        ori_value = source_inputs[aligning_idx]
                        new_value = reject_sample(all_vocab, exception=[inputs[aligning_idx]])
                        source_inputs[aligning_idx] = new_value
                        source_value_maps[aligning_idx] = new_value
                    else:
                        # != True -> False
                        # == False -> True
                        source_inputs[aligning_idx] = inputs[associate_idx]
                        source_value_maps[aligning_idx] = value_maps[associate_idx]
                # 2) we change everything else but not C_{target}
                elif strategy == "softkeep_C":
                    _, source_inputs, _, source_value_maps = sample_factual_inputs(
                        program, all_vocab, synonyms_pairs, synonyms_dict
                    )
                    source_inputs[aligning_idx] = inputs[associate_idx]
                    source_value_maps[aligning_idx] = value_maps[associate_idx]
                elif strategy == "random_C":
                    _, source_inputs, _, source_value_maps = sample_factual_inputs(
                        program, all_vocab, synonyms_pairs, synonyms_dict
                    )
            source_value_maps["op5"] = eval_program(program, source_value_maps, synonyms_pairs, synonyms_dict)
        else:
            _, source_inputs, _, source_value_maps = sample_factual_inputs(
                program, all_vocab, synonyms_pairs, synonyms_dict
            )

        answers = fetch_counterfactual_value(
            value_maps, source_value_maps, program, 
            aligning_causal_variable, "op5",
            all_vocab, synonyms_pairs, synonyms_dict
        )
        
        input_words = [inputs[i] for i in range(len(inputs))]
        input_sentence = ",".join(input_words) 
        source_input_words = [source_inputs[i] for i in range(len(source_inputs))]
        source_input_sentence = ",".join(source_input_words) 
        
        base_answers = value_maps["op5"]
        source_answers = source_value_maps["op5"]
        
        if len(input_output_dict["question"]) < n_sample//2:
            if base_answers == answers:
                continue
        else:
            if base_answers != answers:
                continue        
        
        if input_sentence not in input_output_dict["question"] or \
            source_input_sentence not in input_output_dict["source_question"]:
            input_output_dict["question"] += [input_sentence]
            input_output_dict["source_question"] += [source_input_sentence]
            input_output_dict["answers"] += [answers]
            input_output_dict["intervention_ids"] += [
                aligning_causal_variable_map[aligning_causal_variable]
            ]
            input_output_dict["base_answers"] += [value_maps["op5"]]
            input_output_dict["source_answers"] += [source_value_maps["op5"]]

    return input_output_dict



def make_supervised_counterfactual_data_module(
    program,
    aligning_causal_variable,
    n_alignment_training_examples,
    target_word_beam, 
    token_position_strategy,
    tokenizer,
    test_size=1000
):
    clm_new_token_trigger = "="
    
    all_vocab, synonyms_pairs, synonyms_dict = fetch_metadata(".", use_token=True)
    train_cdataset = Dataset.from_dict(
        prepare_counterfactual_alignment_data_simple(
            program[1],
            n_alignment_training_examples,
            aligning_causal_variable,
            all_vocab, synonyms_pairs, synonyms_dict
        )
    )
    
    validation_cdataset = Dataset.from_dict(
        prepare_counterfactual_alignment_data_simple(
            program[1],
            test_size,
            aligning_causal_variable,
            all_vocab, synonyms_pairs, synonyms_dict
        )
    )
    test_cdataset = Dataset.from_dict(
        prepare_counterfactual_alignment_data_simple(
            program[1],
            test_size,
            aligning_causal_variable,
            all_vocab, synonyms_pairs, synonyms_dict
        )
    )
    
    def counterfactual_preprocess_function(
        target_word_beam,
        token_position_strategy,
        no_answer,
        examples,
    ):
        inputs = examples["question"]
        before_target_inptus = [",".join(_input.split(",")[:target_word_beam]) for _input in inputs]
        target_word_inptus = [_input.split(",")[target_word_beam] for _input in inputs]

        source_inputs = examples["source_question"]
        before_target_source_inptus = [
            ",".join(_input.split(",")[:target_word_beam]) for _input in source_inputs
        ]
        target_word_source_inptus = [
            _input.split(",")[target_word_beam] for _input in source_inputs
        ]

        targets = examples["answers"]

        if no_answer:
            base_examples = [s + f"{clm_new_token_trigger}" for s, t in zip(inputs, targets)]
            source_examples = [s + f"{clm_new_token_trigger}" for s, t in zip(source_inputs, targets)]
        else:
            # We added in a '=' to be the trigger word of answer.
            base_examples = [s + f"{clm_new_token_trigger}" + f"{t}" for s, t in zip(inputs, targets)]
            # note that the target here is a dummy target for both examples.
            # it is the counterfactual target which should not match with these
            # two inputs individually.

            # note that we cancel the eos token as well, as we don't need it.
            source_examples = [s + f"{clm_new_token_trigger}" + f"{t}" for s, t in zip(source_inputs, targets)]

        examples_tokenized = tokenizer(
            base_examples,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        source_examples_tokenized = tokenizer(
            source_examples,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        inputs_tokenized = tokenizer(
            inputs,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        before_target_tokenized = tokenizer(
            before_target_inptus,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        target_word_tokenized = tokenizer(
            target_word_inptus,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        before_target_source_tokenized = tokenizer(
            before_target_source_inptus,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        target_word_source_tokenized = tokenizer(
            target_word_source_inptus,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        intervention_pos = []
        source_intervention_pos = []

        labels = copy.deepcopy(examples_tokenized["input_ids"])

        for i in range(len(inputs_tokenized["input_ids"])):
            input_len = len(inputs_tokenized["input_ids"][i]) + 1 
            # let's not predict the trigger.
            # 1 here is a little hacky... please not follow this.
            labels_t = torch.tensor(labels[i])
            labels_t[:input_len] = IGNORE_INDEX
            labels[i] = labels_t.tolist()

            beam_start_index = len(before_target_tokenized['input_ids'][i])+1
            beam_end_index = beam_start_index + \
                len(target_word_tokenized['input_ids'][i])

            beam_start_source_index = len(before_target_source_tokenized['input_ids'][i])+1
            beam_end_source_index = beam_start_source_index + \
                len(target_word_source_tokenized['input_ids'][i])

            beam_indices = [i for i in range(beam_start_index, beam_end_index)]
            beam_source_indices = [i for i in range(beam_start_source_index, beam_end_source_index)]
            
            if isinstance(token_position_strategy, list):
                intervention_pos += [token_position_strategy]
                source_intervention_pos += [token_position_strategy]
            elif token_position_strategy == "last_of_beam":
                intervention_pos += [[beam_indices[-1], beam_indices[-1]+1]]
                source_intervention_pos += [[beam_source_indices[-1], beam_source_indices[-1]+1]]
            elif token_position_strategy == "first_of_beam":
                intervention_pos += [[beam_indices[0], beam_indices[0]+1]]
                source_intervention_pos += [[beam_source_indices[0], beam_source_indices[0]+1]]
            else:
                assert False, f"Strategy {token_position_strategy} Not Implemented."

        examples_tokenized["source_input_ids"] = source_examples_tokenized["input_ids"]
        examples_tokenized["source_attention_mask"] = source_examples_tokenized["attention_mask"]

        examples_tokenized["labels"] = labels
        examples_tokenized["intervention_ids"] = examples["intervention_ids"]
        # Now, this is the most important part!
        # This is also a novel thing that we introduce in this tutorial, which is for
        # across position interventions.
        examples_tokenized["token_range"] = intervention_pos
        examples_tokenized["source_token_range"] = source_intervention_pos

        return examples_tokenized

    remove_columns=['question', 'source_question', 'answers', 'base_answers', 'source_answers']
    train_cdataset = train_cdataset.map(
        partial(counterfactual_preprocess_function, target_word_beam, token_position_strategy, False),
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=remove_columns,
        desc="Running tokenizer on the train dataset",
    )
    validation_cdataset = validation_cdataset.map(
        partial(counterfactual_preprocess_function, target_word_beam, token_position_strategy, False),
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=remove_columns,
        desc="Running tokenizer on the validation dataset",
    )
    test_cdataset = test_cdataset.map(
        partial(counterfactual_preprocess_function, target_word_beam, token_position_strategy, False),
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
    )

def make_supervised_counterfactual_data_module_single_preload(
    program,
    aligning_causal_variable,
    n_alignment_training_examples,
    target_word_beam, 
    token_position_strategy,
    tokenizer,
    preload_dataset
):
    clm_new_token_trigger = "="
    
    def counterfactual_preprocess_function(
        target_word_beam,
        token_position_strategy,
        no_answer,
        examples,
    ):
        inputs = examples["question"]
        before_target_inptus = [",".join(_input.split(",")[:target_word_beam]) for _input in inputs]
        target_word_inptus = [_input.split(",")[target_word_beam] for _input in inputs]

        source_inputs = examples["source_question"]
        before_target_source_inptus = [
            ",".join(_input.split(",")[:target_word_beam]) for _input in source_inputs
        ]
        target_word_source_inptus = [
            _input.split(",")[target_word_beam] for _input in source_inputs
        ]

        targets = examples["answers"]

        if no_answer:
            base_examples = [s + f"{clm_new_token_trigger}" for s, t in zip(inputs, targets)]
            source_examples = [s + f"{clm_new_token_trigger}" for s, t in zip(source_inputs, targets)]
        else:
            # We added in a '=' to be the trigger word of answer.
            base_examples = [s + f"{clm_new_token_trigger}" + f"{t}" for s, t in zip(inputs, targets)]
            # note that the target here is a dummy target for both examples.
            # it is the counterfactual target which should not match with these
            # two inputs individually.

            # note that we cancel the eos token as well, as we don't need it.
            source_examples = [s + f"{clm_new_token_trigger}" + f"{t}" for s, t in zip(source_inputs, targets)]

        examples_tokenized = tokenizer(
            base_examples,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        source_examples_tokenized = tokenizer(
            source_examples,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        inputs_tokenized = tokenizer(
            inputs,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        before_target_tokenized = tokenizer(
            before_target_inptus,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        target_word_tokenized = tokenizer(
            target_word_inptus,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        before_target_source_tokenized = tokenizer(
            before_target_source_inptus,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        target_word_source_tokenized = tokenizer(
            target_word_source_inptus,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        intervention_pos = []
        source_intervention_pos = []

        labels = copy.deepcopy(examples_tokenized["input_ids"])

        for i in range(len(inputs_tokenized["input_ids"])):
            input_len = len(inputs_tokenized["input_ids"][i]) + 1 
            # let's not predict the trigger.
            # 1 here is a little hacky... please not follow this.
            labels_t = torch.tensor(labels[i])
            labels_t[:input_len] = IGNORE_INDEX
            labels[i] = labels_t.tolist()

            beam_start_index = len(before_target_tokenized['input_ids'][i])+1
            beam_end_index = beam_start_index + \
                len(target_word_tokenized['input_ids'][i])

            beam_start_source_index = len(before_target_source_tokenized['input_ids'][i])+1
            beam_end_source_index = beam_start_source_index + \
                len(target_word_source_tokenized['input_ids'][i])

            beam_indices = [i for i in range(beam_start_index, beam_end_index)]
            beam_source_indices = [i for i in range(beam_start_source_index, beam_end_source_index)]
            
            if isinstance(token_position_strategy, list):
                intervention_pos += [token_position_strategy]
                source_intervention_pos += [token_position_strategy]
            elif token_position_strategy == "last_of_beam":
                intervention_pos += [[beam_indices[-1], beam_indices[-1]+1]]
                source_intervention_pos += [[beam_source_indices[-1], beam_source_indices[-1]+1]]
            elif token_position_strategy == "first_of_beam":
                intervention_pos += [[beam_indices[0], beam_indices[0]+1]]
                source_intervention_pos += [[beam_source_indices[0], beam_source_indices[0]+1]]
            else:
                assert False, f"Strategy {token_position_strategy} Not Implemented."

        examples_tokenized["source_input_ids"] = source_examples_tokenized["input_ids"]
        examples_tokenized["source_attention_mask"] = source_examples_tokenized["attention_mask"]

        examples_tokenized["labels"] = labels
        examples_tokenized["intervention_ids"] = examples["intervention_ids"]
        # Now, this is the most important part!
        # This is also a novel thing that we introduce in this tutorial, which is for
        # across position interventions.
        examples_tokenized["token_range"] = intervention_pos
        examples_tokenized["source_token_range"] = source_intervention_pos

        return examples_tokenized

    remove_columns=['question', 'source_question', 'answers', 'base_answers', 'source_answers']
    preload_dataset = preload_dataset.map(
        partial(counterfactual_preprocess_function, target_word_beam, token_position_strategy, False),
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=remove_columns,
        desc="Running tokenizer on the test dataset",
    )
    
    return preload_dataset

def plot_head_level_attention_heatmaps(attention_data, highlight_heads=[]):
    fig, axs = plt.subplots(3, 4, figsize=(6, 4))  # Adjust size as needed

    # Ensure data has 3 dimensions (num_heads, seq_len, seq_len)
    assert len(attention_data.shape) == 3

    for i in range(attention_data.shape[0]):  # Iterate over attention heads
        ax = axs[i//4, i%4]
        sns.heatmap(attention_data[i], cmap="YlGnBu", square=True, ax=ax, cbar=False)
        ax.set_title(f'Head {i+1}')  # Optionally, set title for each subplot

        # If the head is in the highlight list, change the border color and width
        if i+1 in highlight_heads:
            rect = Rectangle((0,0), 1, 1, fill=False, color='red', lw=3, transform=ax.transAxes, clip_on=False)
            ax.add_patch(rect)

    # If the number of heads is less than 12, remove the empty subplots
    if attention_data.shape[0] < 12:
        for i in range(attention_data.shape[0], 12):
            fig.delaxes(axs.flatten()[i])

    plt.tight_layout()  # Ensure the subplots do not overlap
    plt.show()

    
def test_for_fetch_counterfactual_value(
    program,
    all_vocab, synonyms_pairs, synonyms_dict
):
    def s():
        return random.choice([True, False])
    for _ in range(100000):
        base_value_maps = {
            'op1': s(), 'op2': s(), 
            'op3': s(), 'op4': None, 
            'op5': None
        }
        source_value_maps = {
            'op1': s(), 'op2': s(), 
            'op3': s(), 'op4': None, 
            'op5': None
        }
        eval_l = fetch_counterfactual_value(
            base_value_maps, source_value_maps, program[1], "op1", "op5",
            all_vocab, synonyms_pairs, synonyms_dict
        )
        l = (source_value_maps['op1'] and base_value_maps['op2']) or base_value_maps['op3']
        assert l == eval_l

        eval_l = fetch_counterfactual_value(
            base_value_maps, source_value_maps, program[1], "op2", "op5",
            all_vocab, synonyms_pairs, synonyms_dict
        )
        l = (base_value_maps['op1'] and source_value_maps['op2']) or base_value_maps['op3']
        assert l == eval_l

        eval_l = fetch_counterfactual_value(
            base_value_maps, source_value_maps, program[1], "op3", "op5",
            all_vocab, synonyms_pairs, synonyms_dict
        )
        l = (base_value_maps['op1'] and base_value_maps['op2']) or source_value_maps['op3']
        assert l == eval_l

        eval_l = fetch_counterfactual_value(
            base_value_maps, source_value_maps, program[1], "op4", "op5",
            all_vocab, synonyms_pairs, synonyms_dict
        )
        l = source_value_maps['op4'] or base_value_maps['op3']
        assert l == eval_l

        eval_l = fetch_counterfactual_value(
            base_value_maps, source_value_maps, program[1], "op5", "op5",
            all_vocab, synonyms_pairs, synonyms_dict
        )
        l = source_value_maps['op5']
        assert l == eval_l

    # these are simplier test cases.
    base_value_maps = {0: 'ilitary', 1: 'quickShip', 2: 'ilitary', 3: 'kef', 4: 'Several'}
    source_value_maps = {0: 'ilitary', 1: 'quickShip', 2: 'Several', 3: 'kef', 4: 'kef'}
    assert fetch_counterfactual_value(
        base_value_maps, source_value_maps, program[1], "C2", "op1",
        all_vocab, synonyms_pairs, synonyms_dict
    ) == True

    base_value_maps = {0: 'ilitary', 1: 'quickShip', 2: 'ilitary', 3: 'kef', 4: 'Several'}
    source_value_maps = {0: 'ilitary', 1: 'quickShip', 2: 'Several', 3: 'kef', 4: 'kef'}
    assert fetch_counterfactual_value(
        base_value_maps, source_value_maps, program[1], "C0", "op1",
        all_vocab, synonyms_pairs, synonyms_dict
    ) == False

    base_value_maps = {0: 'ilitary', 1: 'quickShip', 2: 'ilitary', 3: 'kef', 4: 'Several'}
    source_value_maps = {0: 'ilitary', 1: 'quickShip', 2: 'Several', 3: 'kef', 4: 'kef'}
    assert fetch_counterfactual_value(
        base_value_maps, source_value_maps, program[1], "C0", "op2",
        all_vocab, synonyms_pairs, synonyms_dict
    ) == True

    base_value_maps = {0: 'ilitary', 1: 'quickShip', 2: 'ilitary', 3: 'kef', 4: 'Several'}
    source_value_maps = {0: 'Several', 1: 'quickShip', 2: 'Several', 3: 'kef', 4: 'kef'}
    assert fetch_counterfactual_value(
        base_value_maps, source_value_maps, program[1], "C0", "op2",
        all_vocab, synonyms_pairs, synonyms_dict
    ) == False
    print("Tests Passed for fetch_counterfactual_value()")


def is_not_identity_check(program, causal_variable):
    causal_variable_int = int(causal_variable[-1])
    if causal_variable_int in program[0][0]:
        return True
    return False
    
def get_associate_variables(program, causal_variable):
    causal_variable_int = int(causal_variable[-1])
    if causal_variable_int == program[0][0][0]:
        return f"C{program[0][0][1]}"
    elif causal_variable_int == program[0][0][1]:
        return f"C{program[0][0][0]}"
    elif causal_variable_int == program[0][2][0]:
        return f"C{program[0][2][1]}"
    elif causal_variable_int == program[0][2][1]:
        return random.choice([f"C{program[0][2][0]}", f"C{program[0][2][2]}"])
    elif causal_variable_int == program[0][2][2]:
        return f"C{program[0][2][1]}"

def one_good_way_to_prepare_counterfactual_alignment_data(
    program,
    n_sample,
    aligning_causal_variable,
    all_vocab, synonyms_pairs, synonyms_dict
):
    aligning_causal_variable_map = {
        "op1": 0,
        "op2": 0,
        "op3": 0,
        "op4": 0,
        "op5": 0,
        "C0": 0,
        "C1": 0,
        "C2": 0,
        "C3": 0,
        "C4": 0
    }
    ##################################
    #
    # Try to implement this by yourself!
    # You can know more about alignment.
    #
    ##################################
    input_output_dict = {
        "question": [],
        "source_question": [],
        "intervention_ids": [],
        "answers": [],
        "base_answers": [],
        "source_answers": [],
        "alternative_answers": []
    }
    while len(input_output_dict["question"]) < n_sample:
        
        _, inputs, _, value_maps = sample_factual_inputs(
            program, all_vocab, synonyms_pairs, synonyms_dict
        )

        if aligning_causal_variable in {"C0", "C1", "C2", "C3", "C4"}:
            # we use a more systematic intervention sampling strategy.
            strategy = random.choice(["change_C", "softkeep_C"])
            aligning_idx = int(aligning_causal_variable[-1])
            associate_variable = get_associate_variables(program, aligning_causal_variable)
            associate_idx = int(associate_variable[-1])
                        
            # 1) we keep everything else the same but change C_{target}
            if strategy == "change_C":
                source_inputs = copy.deepcopy(inputs)
                source_value_maps = copy.deepcopy(value_maps)
                
                if random.random() < 0.5:
                    # likely != True -> True
                    # likely == True -> False
                    # likely != False -> True
                    # likely == False -> False
                    ori_value = source_inputs[aligning_idx]
                    new_value = reject_sample(all_vocab, exception=[inputs[aligning_idx]])
                    source_inputs[aligning_idx] = new_value
                    source_value_maps[aligning_idx] = new_value
                    
                    answers = fetch_counterfactual_value(
                        value_maps, source_value_maps, program, 
                        aligning_causal_variable, "op5",
                        all_vocab, synonyms_pairs, synonyms_dict
                    )
                    base_answers = value_maps["op5"]
                    
                else:
                    # != True -> False
                    # == False -> True
                    if is_not_identity_check(program, aligning_causal_variable):
                        # semantic soft retrieve.
                        new_val = random.choice(synonyms_dict[inputs[associate_idx]])
                        source_inputs[aligning_idx] = new_val
                        source_value_maps[aligning_idx] = new_val
                    else:
                        source_inputs[aligning_idx] = inputs[associate_idx]
                        source_value_maps[aligning_idx] = value_maps[associate_idx]
                        
            elif strategy == "softkeep_C":
            # 2) we change everything else but not C_{target}
                _, source_inputs, _, source_value_maps = sample_factual_inputs(
                    program, all_vocab, synonyms_pairs, synonyms_dict
                )
                if random.random() < 0.5:
                    # no change label!
                    source_inputs[aligning_idx] = inputs[aligning_idx]
                    source_value_maps[aligning_idx] = value_maps[aligning_idx]
                else:
                    if is_not_identity_check(program, aligning_causal_variable):
                        # semantic soft retrieve.
                        new_val = random.choice(synonyms_dict[inputs[associate_idx]])
                        source_inputs[aligning_idx] = new_val
                        source_value_maps[aligning_idx] = new_val
                    else:
                        source_inputs[aligning_idx] = inputs[associate_idx]
                        source_value_maps[aligning_idx] = value_maps[associate_idx]
                        
            source_value_maps["op5"] = eval_program(program, source_value_maps, synonyms_pairs, synonyms_dict)
        else:
            _, source_inputs, _, source_value_maps = sample_factual_inputs(
                program, all_vocab, synonyms_pairs, synonyms_dict
            )

        answers = fetch_counterfactual_value(
            value_maps, source_value_maps, program, 
            aligning_causal_variable, "op5",
            all_vocab, synonyms_pairs, synonyms_dict
        )
        
        input_words = [inputs[i] for i in range(len(inputs))]
        input_sentence = ",".join(input_words) 
        source_input_words = [source_inputs[i] for i in range(len(source_inputs))]
        source_input_sentence = ",".join(source_input_words) 
        
        base_answers = value_maps["op5"]
        source_answers = source_value_maps["op5"]
        
        if len(input_output_dict["question"]) < n_sample//2:
            if base_answers == answers:
                continue
        else:
            if base_answers != answers:
                continue
        if input_sentence not in input_output_dict["question"] or \
            source_input_sentence not in input_output_dict["source_question"]:
            input_output_dict["question"] += [input_sentence]
            input_output_dict["source_question"] += [source_input_sentence]
            input_output_dict["answers"] += [answers]
            input_output_dict["intervention_ids"] += [
                aligning_causal_variable_map[aligning_causal_variable]
            ]
            input_output_dict["base_answers"] += [value_maps["op5"]]
            input_output_dict["source_answers"] += [source_value_maps["op5"]]
            # we want to get alternative answers as if we are intervening
            # on different causal variables. again, if some variables have
            # the exact same ctf output, we cannot distinguish them!
            all_alternative_answers = {}
            for alternative_var in {"op1", "op2", "op3", "op4", "op5", "C0", "C1", "C2", "C3", "C4"}:
                alternative_answers = fetch_counterfactual_value(
                    value_maps, source_value_maps, program, 
                    alternative_var, "op5",
                    all_vocab, synonyms_pairs, synonyms_dict
                )
                all_alternative_answers[alternative_var] = alternative_answers
            input_output_dict["alternative_answers"] += [all_alternative_answers]

    return input_output_dict

    
def bad_way_to_prepare_counterfactual_alignment_data(
    program,
    n_sample,
    aligning_causal_variable,
    all_vocab, synonyms_pairs, synonyms_dict
):
    aligning_causal_variable_map = {
        "op1": 0,
        "op2": 0,
        "op3": 0,
        "op4": 0,
        "op5": 0,
        "C0": 0,
        "C1": 0,
        "C2": 0,
        "C3": 0,
        "C4": 0
    }
    ##################################
    #
    # Try to implement this by yourself!
    # You can know more about alignment.
    #
    ##################################
    input_output_dict = {
        "question": [],
        "source_question": [],
        "intervention_ids": [],
        "answers": [],
        "base_answers": [],
        "source_answers": [],
        "alternative_answers": []
    }
    while len(input_output_dict["question"]) < n_sample:
        
        _, inputs, _, value_maps = sample_factual_inputs(
            program, all_vocab, synonyms_pairs, synonyms_dict
        )
        _, source_inputs, _, source_value_maps = sample_factual_inputs(
            program, all_vocab, synonyms_pairs, synonyms_dict
        )
        if aligning_causal_variable in {"C0", "C1", "C2", "C3", "C4"}:
            if random.random() < 0.5:
                # we replace the aligning input token with
                # the one from base to simulate a null intervention.
                # as a result, base input should not be changed.
                source_inputs[int(aligning_causal_variable[-1])] = inputs[int(aligning_causal_variable[-1])]
                source_value_maps[int(aligning_causal_variable[-1])] = value_maps[int(aligning_causal_variable[-1])]
                source_value_maps["op5"] = eval_program(program, source_value_maps, synonyms_pairs, synonyms_dict)

        answers = fetch_counterfactual_value(
            value_maps, source_value_maps, program, 
            aligning_causal_variable, "op5",
            all_vocab, synonyms_pairs, synonyms_dict
        )
        
        input_words = [inputs[i] for i in range(len(inputs))]
        input_sentence = ",".join(input_words) 
        source_input_words = [source_inputs[i] for i in range(len(source_inputs))]
        source_input_sentence = ",".join(source_input_words) 
        
        base_answers = value_maps["op5"]
        source_answers = source_value_maps["op5"]
        
        if len(input_output_dict["question"]) < n_sample//2:
            if base_answers == answers:
                continue
        else:
            if base_answers != answers:
                continue
        if input_sentence not in input_output_dict["question"] or \
            source_input_sentence not in input_output_dict["source_question"]:
            input_output_dict["question"] += [input_sentence]
            input_output_dict["source_question"] += [source_input_sentence]
            input_output_dict["answers"] += [answers]
            input_output_dict["intervention_ids"] += [
                aligning_causal_variable_map[aligning_causal_variable]
            ]
            input_output_dict["base_answers"] += [value_maps["op5"]]
            input_output_dict["source_answers"] += [source_value_maps["op5"]]
            # we want to get alternative answers as if we are intervening
            # on different causal variables. again, if some variables have
            # the exact same ctf output, we cannot distinguish them!
            all_alternative_answers = {}
            for alternative_var in {"op1", "op2", "op3", "op4", "op5", "C0", "C1", "C2", "C3", "C4"}:
                alternative_answers = fetch_counterfactual_value(
                    value_maps, source_value_maps, program, 
                    alternative_var, "op5",
                    all_vocab, synonyms_pairs, synonyms_dict
                )
                all_alternative_answers[alternative_var] = alternative_answers
            input_output_dict["alternative_answers"] += [all_alternative_answers]
            
    return input_output_dict

def plot_single_transformer(
    ax, max_length=10, num_layers=12, highlights=None, xtick_labels=None,
    vmin=0.80, vmax=1.00, title="Transformer Block View (residual+mlp)"
):
    data = np.zeros((num_layers, max_length))
    
    # Add highlights
    if highlights is not None:
        for highlight in highlights:
            data[highlight[0], highlight[1]] = highlight[2]

    # Plot the heatmap
    sns.heatmap(
        data, annot=True, linewidths=1,
        cmap="cividis", cbar=False, 
        vmin=vmin, vmax=vmax, # this is rather abitrary to help visualizing!
        ax=ax
    )
    
    # Customize the plot
    ax.set_xlabel('Token')
    ax.set_ylabel('Layer')
    ax.set_title(title)
     
    # Set aspect ratio
    ax.set_aspect('0.5')
    
    # Set x-tick labels if provided
    if xtick_labels is not None:
        ax.set_xticklabels(xtick_labels)
    
    ax.invert_yaxis()