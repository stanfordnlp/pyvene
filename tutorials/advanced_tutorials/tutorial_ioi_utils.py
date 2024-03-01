import math
import joblib
import os
import inspect
from tqdm import tqdm
from collections import defaultdict
import functools
from collections import OrderedDict
from abc import ABC, abstractmethod
import json
from pathlib import Path
import random
from typing import (
    Tuple,
    List,
    Sequence,
    Union,
    Any,
    Optional,
    Literal,
    Iterable,
    Callable,
    Dict,
)
import typing
import transformers

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import Parameter
from torch import nn
from torch.utils.data import Dataset, DataLoader

from pyvene import (
    IntervenableModel,
    RepresentationConfig,
    IntervenableConfig,
    LowRankRotatedSpaceIntervention,
    SkipIntervention,
    VanillaIntervention,
    BoundlessRotatedSpaceIntervention,
)

###
#
# Dataset generation code is mostly copied from
# https://github.com/amakelov/activation-patching-illusion
#
###


def is_single_token(s: str, tokenizer) -> bool:
    """
    Check if a string is a single token in the vocabulary of a model.
    """
    return len(tokenizer.tokenize(s)) == 1


NAMES_PATH = "tutorial_data/names.json"
OBJECTS_PATH = "tutorial_data/objects.json"
PLACES_PATH = "tutorial_data/places.json"
TEMPLATES_PATH = "tutorial_data/templates.json"

NAMES = json.load(open(NAMES_PATH))
OBJECTS = json.load(open(OBJECTS_PATH))
PLACES = json.load(open(PLACES_PATH))
TEMPLATES = json.load(open(TEMPLATES_PATH))


class Prompt:
    """
    Represent a general ABC prompt using a template, and operations on it that
    are useful for generating datasets.
    """

    def __init__(
        self,
        names: Tuple[str, str, str],
        template: str,
        obj: str,
        place: str,
    ):
        self.names = names
        self.template = template
        self.obj = obj
        self.place = place
        if self.is_ioi:
            self.s_name = self.names[2]  # subject always appears in third position
            self.io_name = [x for x in self.names[:2] if x != self.s_name][0]
            self.s1_pos = self.names[:2].index(self.s_name)
            self.io_pos = self.names[:2].index(self.io_name)
            self.s2_pos = 2
        else:
            self.io_name = None
            self.s_name = None

    @property
    def is_ioi(self) -> bool:
        return self.names[2] in self.names[:2] and len(set(self.names)) == 2

    def __repr__(self) -> str:
        return f"<===PROMPT=== {self.sentence}>"

    @property
    def sentence(self) -> str:
        return self.template.format(
            name_A=self.names[0],
            name_B=self.names[1],
            name_C=self.names[2],
            object=self.obj,
            place=self.place,
        )

    @staticmethod
    def canonicalize(things: Tuple[str, str, str]) -> Tuple[str, str, str]:
        # the unique elements of the tuple, in the order they appear
        ordered_uniques = list(OrderedDict.fromkeys(things).keys())
        canonical_elts = ["A", "B", "C"]
        uniques_to_canonical = {
            x: y
            for x, y in zip(ordered_uniques, canonical_elts[: len(ordered_uniques)])
        }
        return tuple([uniques_to_canonical[x] for x in things])

    @staticmethod
    def matches_pattern(names: Tuple[str, str, str], pattern: str) -> bool:
        return Prompt.canonicalize(names) == Prompt.canonicalize(tuple(pattern))

    def resample_pattern(
        self, orig_pattern: str, new_pattern: str, name_distribution: Sequence[str]
    ) -> "Prompt":
        """
        Change the pattern of the prompt, while keeping the names that are
        mapped to the same symbols in the original and new patterns the same.

        Args:
            orig_pattern (str): _description_
            new_pattern (str): _description_
            name_distribution (Sequence[str]): _description_

        Example:
            prompt = train_distribution.sample_one(pattern='ABB')
            (prompt.sentence,
            prompt.resample_pattern(orig_pattern='ABB', new_pattern='BAA',
                                    name_distribution=train_distribution.names,).sentence,
            prompt.resample_pattern(orig_pattern='ABB', new_pattern='CDD',
                                    name_distribution=train_distribution.names,).sentence,
            prompt.resample_pattern(orig_pattern='ABB', new_pattern='ACC',
                                    name_distribution=train_distribution.names,).sentence,

        >>> ('Then, Olivia and Anna had a long and really crazy argument. Afterwards, Anna said to',
        >>> 'Then, Anna and Olivia had a long and really crazy argument. Afterwards, Olivia said to',
        >>> 'Then, Joe and Kelly had a long and really crazy argument. Afterwards, Kelly said to',
        >>> 'Then, Olivia and Carl had a long and really crazy argument. Afterwards, Carl said to')
        )
        """
        assert len(orig_pattern) == 3
        assert len(new_pattern) == 3
        assert self.matches_pattern(names=self.names, pattern=orig_pattern)
        orig_to_name = {orig_pattern[i]: self.names[i] for i in range(3)}
        new_names = [None for _ in range(3)]
        new_pos_to_symbol = {}
        for i, symbol in enumerate(new_pattern):
            if symbol in orig_to_name.keys():
                new_names[i] = orig_to_name[symbol]
            else:
                new_pos_to_symbol[i] = symbol
        new_symbols = new_pos_to_symbol.values()
        if len(new_symbols) > 0:
            new_symbol_to_name = {}
            # must sample some *new* names
            available_names = [x for x in name_distribution if x not in self.names]
            for symbol in new_symbols:
                new_symbol_to_name[symbol] = random.choice(available_names)
                available_names.remove(new_symbol_to_name[symbol])
            # populate new_names with new symbols
            for i, symbol in new_pos_to_symbol.items():
                new_names[i] = new_symbol_to_name[symbol]
        return Prompt(
            names=tuple(new_names),
            template=self.template,
            obj=self.obj,
            place=self.place,
        )


def load_data(data: Union[List[str], str, Path]) -> List[str]:
    if isinstance(data, (str, Path)):
        with open(data) as f:
            data: List[str] = json.load(f)
    return data


class PromptDataset(Dataset):
    def __init__(self, prompts: List[Prompt], tokenizer):
        assert len(prompts) > 0
        self.prompts: Sequence[Prompt] = np.array(prompts)
        self.tokenizer = tokenizer
        ls = self.lengths

    def __getitem__(self, idx: Union[int, Sequence, slice]) -> "PromptDataset":
        if isinstance(idx, int):
            prompts = [self.prompts[idx]]
        else:
            prompts = self.prompts[idx]
            if isinstance(prompts, Prompt):
                prompts = [prompts]
        assert all(isinstance(x, Prompt) for x in prompts)
        return PromptDataset(prompts=prompts, tokenizer=self.tokenizer)

    def __len__(self) -> int:
        return len(self.prompts)

    def __repr__(self) -> str:
        return f"{[x for x in self.prompts]}"

    def __add__(self, other: "PromptDataset") -> "PromptDataset":
        return PromptDataset(
            prompts=list(self.prompts) + list(other.prompts), tokenizer=self.tokenizer
        )

    @property
    def lengths(self) -> List[int]:
        return [
            self.tokenizer(
                x.sentence, return_tensors="pt", return_attention_mask=False
            )["input_ids"].shape[1]
            for x in self.prompts
        ]

    @property
    def tokens(self) -> Tensor:
        return self.tokenizer(
            [x.sentence for x in self.prompts],
            return_tensors="pt",
            padding=True,
        )

    @property
    def io_tokens(self) -> Tensor:
        return torch.tensor(
            [self.tokenizer(f" {x.io_name}")["input_ids"][0] for x in self.prompts]
        )

    @property
    def s_tokens(self) -> Tensor:
        return torch.tensor(
            [self.tokenizer(f" {x.s_name}")["input_ids"][0] for x in self.prompts]
        )

    @property
    def answer_tokens(self):
        # return a tensor with two columns: self.io_tokens and self.s_tokens
        return torch.tensor(
            [
                [
                    self.tokenizer(f" {x.io_name}")["input_ids"][0],
                    self.tokenizer(f" {x.s_name}")["input_ids"][0],
                ]
                for x in self.prompts
            ]
        )


def get_last_token(logits, attention_mask):
    last_token_indices = attention_mask.sum(1) - 1
    batch_indices = torch.arange(logits.size(0)).unsqueeze(1)
    return logits[batch_indices, last_token_indices.unsqueeze(1)].squeeze(1)


class PatchingDataset(Dataset):
    """
    Bundle together the data needed to *train* a (DAS or other) patching for a
    single causal variable (we can generalize this later if we need).

    All you need to do *trainable* patching is the base and source
    `PromptDataset`s, and the patched answer tokens of shape (batch, 2), where
        - the 1st column is the patched answer,
        - and the 2nd column is the other possible answer (useful for computing
        logit diffs).

    Since this dataset holds only the bare minimum necessary for patching, it
    decouples the kind of patching we do from the data representation, allowing
    us to treat data in the same way regardless of whether we're doing DAS or
    some other kind of patching.
    """

    def __init__(
        self,
        base: PromptDataset,
        source: PromptDataset,
        patched_answer_tokens,
    ):
        assert len(base) == len(source)
        assert len(base) == len(patched_answer_tokens)
        self.base = base
        self.source = source
        self.patched_answer_tokens = patched_answer_tokens.long()

    def batches(
        self, batch_size: int, shuffle: bool = True
    ) -> Iterable["PatchingDataset"]:
        if shuffle:
            order = np.random.permutation(len(self))
        else:
            order = np.arange(len(self))
        for i in range(0, len(self), batch_size):
            yield self[order[i : i + batch_size]]

    def __getitem__(self, idx) -> "PatchingDataset":
        patched_answer_tokens = self.patched_answer_tokens[idx]
        if len(patched_answer_tokens.shape) == 1:
            patched_answer_tokens = patched_answer_tokens.unsqueeze(0)
        return PatchingDataset(
            base=self.base[idx],
            source=self.source[idx],
            patched_answer_tokens=patched_answer_tokens,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __add__(self, other: "PatchingDataset") -> "PatchingDataset":
        return PatchingDataset(
            base=self.base + other.base,
            source=self.source + other.source,
            patched_answer_tokens=torch.cat(
                [self.patched_answer_tokens, other.patched_answer_tokens], dim=0
            ),
        )


class PromptDistribution:
    """
    A class to represent a distribution over prompts.

    It uses a combination of names, places, objects, and templates
    loaded from JSON files or provided lists.

    Each prompt is constructed using a selected template and a randomly selected
    name, object, and place.
    """

    def __init__(
        self,
        names: Union[List[str], str, Path],
        places: Union[List[str], str, Path],
        objects: Union[List[str], str, Path],
        templates: Union[List[str], str, Path],
    ):
        self.names = load_data(names)
        self.places = load_data(places)
        self.objects = load_data(objects)
        self.templates = load_data(templates)

    def sample_one(
        self,
        pattern: str,
    ) -> Prompt:
        """
        Sample a single prompt from the distribution.
        """
        template = random.choice(self.templates)
        unique_ids = list(set(pattern))
        unique_names = random.sample(self.names, len(unique_ids))
        assert len(set(unique_names)) == len(unique_names)
        prompt_names = tuple([unique_names[unique_ids.index(i)] for i in pattern])
        obj = random.choice(self.objects)
        place = random.choice(self.places)
        return Prompt(
            names=prompt_names, template=template, obj=obj, place=place
        )

    def sample_das(
        self,
        tokenizer,
        base_patterns: List[str],
        source_patterns: List[str],
        samples_per_combination: int,
        labels: Literal["position", "name"],
    ) -> PatchingDataset:
        """
        This samples a dataset of base and corrupted prompts for doing DAS on
        position or name subspaces.

        tokenizer
        samples_per_combination : int
            The number of samples to be generated for each combination of patterns.
        orig_patterns : List[str]
            A list of original patterns that will be used to create the prompts. For example ["ABB", "BAB"].
        corrupted_patterns : List[str]
            A list of corrupted patterns that will be used to create the corrupted prompts.
            Use same letters as in orig_patterns if you want to use the same names, objects, and places.
            Use different letters like ["CDD", "DCD"] if you want to use different names, objects, and places.
        labels : str
            The type of label for the task. Supports 'position' and 'name'.
            The label is the answer token that the model should predict if the position or name information is patched
            into activations during the forward pass of the clean prompt.
        """
        base_prompts: List[Prompt] = []
        source_prompts: List[Prompt] = []
        for orig_pattern in base_patterns:
            for corrupted_pattern in source_patterns:
                base_prompt_batch = [
                    self.sample_one(orig_pattern)
                    for _ in range(samples_per_combination)
                ]
                source_prompt_batch = [
                    p.resample_pattern(
                        name_distribution=self.names,
                        orig_pattern=orig_pattern,
                        new_pattern=corrupted_pattern,
                    )
                    for p in base_prompt_batch
                ]
                base_prompts.extend(base_prompt_batch)
                source_prompts.extend(source_prompt_batch)

        # if DAS should find the position subspace
        if labels == "position":
            patched_answer_names = []  # list of (correct, incorrect) name pairs
            for base_prompt, source_prompt in zip(base_prompts, source_prompts):
                if base_prompt.s1_pos == source_prompt.s1_pos:
                    patched_answer_names.append(
                        (base_prompt.io_name, base_prompt.s_name)
                    )
                else:
                    patched_answer_names.append(
                        (base_prompt.s_name, base_prompt.io_name)
                    )
        elif labels == "name":
            patched_answer_names = []  # list of (correct, incorrect) name pairs
            for base_prompt, source_prompt in zip(base_prompts, source_prompts):
                patched_answer_names.append(
                    (source_prompt.io_name, base_prompt.io_name)
                )

        clean_dataset = PromptDataset(base_prompts, tokenizer)
        corrupted_dataset = PromptDataset(source_prompts, tokenizer)
        patched_answer_tokens = torch.Tensor(
            [
                [
                    tokenizer(f" {x}")["input_ids"][0] for x in y
                ]  # prepend space for each name
                for y in patched_answer_names
            ]
        )
        return PatchingDataset(
            base=clean_dataset,
            source=corrupted_dataset,
            patched_answer_tokens=patched_answer_tokens,
        )


criterion = torch.nn.CrossEntropyLoss()


# You can define your custom compute_metrics function.
def compute_metrics(eval_preds, eval_labels):
    total_count = 0
    correct_count = 0
    kl_divs = []
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        # acc
        actual_test_labels = eval_label
        pred_test_labels = torch.argmax(eval_pred[:, -1], dim=-1)
        correct_labels = actual_test_labels == pred_test_labels
        total_count += len(correct_labels)
        correct_count += correct_labels.sum().tolist()
        # kl div
        kl_divs += [
            eval_pred[:, -1][torch.arange(len(actual_test_labels)), actual_test_labels]
        ]
    accuracy = round(correct_count / total_count, 2)
    kl_div = torch.cat(kl_divs, dim=0).mean()
    return {"accuracy": accuracy, "kl_div": kl_div}


def calculate_loss(logits, labels):
    shift_logits = logits[..., -1, :].contiguous()
    shift_labels = labels.contiguous()
    # Flatten the tokens
    shift_logits = shift_logits
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = criterion(shift_logits, shift_labels)

    return loss


def single_d_low_rank_das_position_config(
    model_type,
    intervention_type,
    layer,
    intervention_types,
    low_rank_dimension=1,
    num_unit=1,
    head_level=False,
):
    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            RepresentationConfig(
                layer,  # layer
                intervention_type,  # intervention type
                "pos" if not head_level else "h.pos",
                num_unit,
                low_rank_dimension=low_rank_dimension,  # a single das direction
            ),
        ],
        intervention_types=intervention_types,
    )
    return config


def calculate_boundless_das_loss(logits, labels, intervenable):
    loss = calculate_loss(logits, labels)
    for k, v in intervenable.interventions.items():
        boundary_loss = 2.0 * v[0].intervention_boundaries.sum()
    loss += boundary_loss
    return loss


def find_variable_at(
    gpt2,
    tokenizer,
    positions,
    layers,
    stream,
    heads=None,
    low_rank_dimension=1,
    aligning_variable="position",
    do_vanilla_intervention=False,
    do_boundless_das=False,
    seed=42,
    return_intervenable=False,
    debug=False,
):
    transformers.set_seed(seed)

    if aligning_variable == "name":
        # we hacky the distribution a little
        train_distribution = PromptDistribution(
            names=NAMES[:20],
            objects=OBJECTS[: len(OBJECTS) // 2],
            places=PLACES[: len(PLACES) // 2],
            templates=TEMPLATES[:2],
        )

        test_distribution = PromptDistribution(
            names=NAMES[:20],
            objects=OBJECTS[len(OBJECTS) // 2 :],
            places=PLACES[len(PLACES) // 2 :],
            templates=TEMPLATES[2:],
        )
    else:
        train_distribution = PromptDistribution(
            names=NAMES[: len(NAMES) // 2],
            objects=OBJECTS[: len(OBJECTS) // 2],
            places=PLACES[: len(PLACES) // 2],
            templates=TEMPLATES[:2],
        )

        test_distribution = PromptDistribution(
            names=NAMES[len(NAMES) // 2 :],
            objects=OBJECTS[len(OBJECTS) // 2 :],
            places=PLACES[len(PLACES) // 2 :],
            templates=TEMPLATES[2:],
        )

    D_train = train_distribution.sample_das(
        tokenizer=tokenizer,
        base_patterns=["ABB", "BAB"],
        source_patterns=["ABB", "BAB"]
        if aligning_variable == "position"
        else ["CDD", "DCD"],
        labels=aligning_variable,
        samples_per_combination=50 if aligning_variable == "position" else 50,
    )
    D_test = test_distribution.sample_das(
        tokenizer=tokenizer,
        base_patterns=[
            "ABB",
        ],
        source_patterns=["BAB"] if aligning_variable == "position" else ["DCD"],
        labels=aligning_variable,
        samples_per_combination=50,
    ) + test_distribution.sample_das(
        tokenizer=tokenizer,
        base_patterns=[
            "BAB",
        ],
        source_patterns=["ABB"] if aligning_variable == "position" else ["CDD"],
        labels=aligning_variable,
        samples_per_combination=50,
    )

    across_positions = True if isinstance(positions[0], list) else False

    data = []

    batch_size = 20
    eval_every = 5
    initial_lr = 0.01
    n_epochs = 10
    aligning_stream = stream

    if do_boundless_das:
        _intervention_type = BoundlessRotatedSpaceIntervention
    elif do_vanilla_intervention:
        _intervention_type = VanillaIntervention
    else:
        _intervention_type = LowRankRotatedSpaceIntervention

    for aligning_pos in positions:
        for aligning_layer in layers:
            if debug:
                print(
                    f"finding name position at: pos->{aligning_pos}, "
                    f"layers->{aligning_layer}, stream->{stream}"
                )
            if heads is not None:
                config = single_d_low_rank_das_position_config(
                    type(gpt2),
                    aligning_stream,
                    aligning_layer,
                    _intervention_type,
                    low_rank_dimension=low_rank_dimension,
                    num_unit=len(heads),
                    head_level=True,
                )
            else:
                if across_positions:
                    config = single_d_low_rank_das_position_config(
                        type(gpt2),
                        aligning_stream,
                        aligning_layer,
                        _intervention_type,
                        low_rank_dimension=low_rank_dimension,
                        num_unit=len(positions[0]),
                    )
                else:
                    config = single_d_low_rank_das_position_config(
                        type(gpt2),
                        aligning_stream,
                        aligning_layer,
                        _intervention_type,
                        low_rank_dimension=low_rank_dimension,
                    )
            intervenable = IntervenableModel(config, gpt2)
            intervenable.set_device("cuda" if torch.cuda.is_available() else "cpu")
            intervenable.disable_model_gradients()
            total_step = 0
            if not do_vanilla_intervention:
                if do_boundless_das:
                    optimizer_params = []
                    for k, v in intervenable.interventions.items():
                        optimizer_params += [{"params": v[0].rotate_layer.parameters()}]
                        optimizer_params += [
                            {"params": v[0].intervention_boundaries, "lr": 0.5}
                        ]
                    optimizer = torch.optim.Adam(optimizer_params, lr=initial_lr)
                    target_total_step = int(len(D_train) / batch_size) * n_epochs
                    temperature_start = 50.0
                    temperature_end = 0.1
                    temperature_schedule = (
                        torch.linspace(
                            temperature_start, temperature_end, target_total_step
                        )
                        .to(torch.bfloat16)
                        .to("cuda" if torch.cuda.is_available() else "cpu")
                    )
                    intervenable.set_temperature(temperature_schedule[total_step])
                else:
                    optimizer = torch.optim.Adam(
                        intervenable.get_trainable_parameters(), lr=initial_lr
                    )
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, end_factor=0.1, total_iters=n_epochs
                )

                for epoch in range(n_epochs):
                    torch.cuda.empty_cache()
                    for batch_dataset in D_train.batches(batch_size=batch_size):
                        # prepare base
                        base_inputs = batch_dataset.base.tokens
                        b_s = base_inputs["input_ids"].shape[0]
                        for k, v in base_inputs.items():
                            if v is not None and isinstance(v, torch.Tensor):
                                base_inputs[k] = v.to(gpt2.device)
                        # prepare source
                        source_inputs = batch_dataset.source.tokens
                        for k, v in source_inputs.items():
                            if v is not None and isinstance(v, torch.Tensor):
                                source_inputs[k] = v.to(gpt2.device)
                        # prepare label
                        labels = batch_dataset.patched_answer_tokens[:, 0].to(
                            gpt2.device
                        )

                        assert all(x == 18 for x in batch_dataset.base.lengths)
                        assert all(x == 18 for x in batch_dataset.source.lengths)

                        if heads is not None:
                            _, counterfactual_outputs = intervenable(
                                {"input_ids": base_inputs["input_ids"]},
                                [{"input_ids": source_inputs["input_ids"]}],
                                {
                                    "sources->base": (
                                        [[[heads] * b_s, [[aligning_pos]] * b_s]],
                                        [[[heads] * b_s, [[aligning_pos]] * b_s]],
                                    )
                                },
                            )
                        else:
                            if across_positions:
                                _, counterfactual_outputs = intervenable(
                                    {"input_ids": base_inputs["input_ids"]},
                                    [{"input_ids": source_inputs["input_ids"]}],
                                    {
                                        "sources->base": (
                                            [[aligning_pos] * b_s],
                                            [[aligning_pos] * b_s],
                                        )
                                    },
                                )
                            else:
                                _, counterfactual_outputs = intervenable(
                                    {"input_ids": base_inputs["input_ids"]},
                                    [{"input_ids": source_inputs["input_ids"]}],
                                    {
                                        "sources->base": (
                                            [[[aligning_pos]] * b_s],
                                            [[[aligning_pos]] * b_s],
                                        )
                                    },
                                )

                        logits = get_last_token(counterfactual_outputs.logits, base_inputs["attention_mask"]).unsqueeze(1)
                        eval_metrics = compute_metrics(
                            [logits], [labels]
                        )
                        if do_boundless_das:
                            loss = calculate_boundless_das_loss(
                                logits, labels, intervenable
                            )
                        else:
                            loss = calculate_loss(logits, labels)
                        loss_str = round(loss.item(), 2)
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        intervenable.set_zero_grad()
                        if do_boundless_das:
                            intervenable.set_temperature(
                                temperature_schedule[total_step]
                            )
                            for k, v in intervenable.interventions.items():
                                intervention_boundaries = v[
                                    0
                                ].intervention_boundaries.sum()
                        total_step += 1

            # eval
            eval_labels = []
            eval_preds = []
            with torch.no_grad():
                torch.cuda.empty_cache()
                for batch_dataset in D_test.batches(batch_size=batch_size):
                    # prepare base
                    base_inputs = batch_dataset.base.tokens
                    b_s = base_inputs["input_ids"].shape[0]
                    for k, v in base_inputs.items():
                        if v is not None and isinstance(v, torch.Tensor):
                            base_inputs[k] = v.to(gpt2.device)
                    # prepare source
                    source_inputs = batch_dataset.source.tokens
                    for k, v in source_inputs.items():
                        if v is not None and isinstance(v, torch.Tensor):
                            source_inputs[k] = v.to(gpt2.device)
                    # prepare label
                    labels = batch_dataset.patched_answer_tokens[:, 0].to(gpt2.device)

                    assert all(x == 18 for x in batch_dataset.base.lengths)
                    assert all(x == 18 for x in batch_dataset.source.lengths)

                    if heads is not None:
                        _, counterfactual_outputs = intervenable(
                            {"input_ids": base_inputs["input_ids"]},
                            [{"input_ids": source_inputs["input_ids"]}],
                            {
                                "sources->base": (
                                    [[[heads] * b_s, [[aligning_pos]] * b_s]],
                                    [[[heads] * b_s, [[aligning_pos]] * b_s]],
                                )
                            },
                        )
                    else:
                        if across_positions:
                            _, counterfactual_outputs = intervenable(
                                {"input_ids": base_inputs["input_ids"]},
                                [{"input_ids": source_inputs["input_ids"]}],
                                {
                                    "sources->base": (
                                        [[aligning_pos] * b_s],
                                        [[aligning_pos] * b_s],
                                    )
                                },
                            )
                        else:
                            _, counterfactual_outputs = intervenable(
                                {"input_ids": base_inputs["input_ids"]},
                                [{"input_ids": source_inputs["input_ids"]}],
                                {
                                    "sources->base": (
                                        [[[aligning_pos]] * b_s],
                                        [[[aligning_pos]] * b_s],
                                    )
                                },
                            )
                    eval_labels += [labels]
                    logits = get_last_token(counterfactual_outputs.logits, base_inputs["attention_mask"]).unsqueeze(1)
                    eval_preds += [logits]
            eval_metrics = compute_metrics(eval_preds, eval_labels)

            if do_boundless_das:
                for k, v in intervenable.interventions.items():
                    intervention_boundaries = v[0].intervention_boundaries.sum()
                data.append(
                    {
                        "pos": aligning_pos,
                        "layer": aligning_layer,
                        "acc": eval_metrics["accuracy"],
                        "kl_div": eval_metrics["kl_div"],
                        "boundary": intervention_boundaries,
                        "stream": stream,
                    }
                )
            else:
                if heads is not None:
                    heads_str = ",".join([str(h) for h in heads])
                    data.append(
                        {
                            "pos": aligning_pos,
                            "layer": aligning_layer,
                            "acc": eval_metrics["accuracy"],
                            "kl_div": eval_metrics["kl_div"],
                            "stream": f"{stream}_{heads_str}",
                        }
                    )
                else:
                    data.append(
                        {
                            "pos": aligning_pos,
                            "layer": aligning_layer,
                            "acc": eval_metrics["accuracy"],
                            "kl_div": eval_metrics["kl_div"],
                            "stream": stream,
                        }
                    )
    if return_intervenable:
        return data, intervenable
    return data


def path_patching_config(
    layer, last_layer, low_rank_dimension,
    component="attention_output", unit="pos"
):
    intervening_component = [{
        "layer": layer, "component": component, 
        "unit": unit, "group_key": 0,
        "intervention_type": LowRankRotatedSpaceIntervention,
        "low_rank_dimension": low_rank_dimension,
    }]
    restoring_components = []
    if not component.startswith("mlp_"):
        restoring_components += [{
            "layer": layer, "component": "mlp_output", "group_key": 1,
            "intervention_type": VanillaIntervention,
        }]
    for i in range(layer+1, last_layer):
        restoring_components += [{
            "layer": i, "component": "attention_output", "group_key": 1, 
            "intervention_type": VanillaIntervention},{
            "layer": i, "component": "mlp_output", "group_key": 1,
            "intervention_type": VanillaIntervention
        }]
    intervenable_config = IntervenableConfig(
        intervening_component + restoring_components)
    return intervenable_config, len(restoring_components)


def with_path_patch_find_variable_at(
    gpt2,
    tokenizer,
    positions,
    layers,
    stream,
    low_rank_dimension=1,
    seed=42,
    debug=False,
):
    transformers.set_seed(seed)

    train_distribution = PromptDistribution(
        names=NAMES[: len(NAMES) // 2],
        objects=OBJECTS[: len(OBJECTS) // 2],
        places=PLACES[: len(PLACES) // 2],
        templates=TEMPLATES[:2],
    )

    test_distribution = PromptDistribution(
        names=NAMES[len(NAMES) // 2 :],
        objects=OBJECTS[len(OBJECTS) // 2 :],
        places=PLACES[len(PLACES) // 2 :],
        templates=TEMPLATES[2:],
    )

    D_train = train_distribution.sample_das(
        tokenizer=tokenizer,
        base_patterns=["ABB", "BAB"],
        source_patterns=["ABB", "BAB"],
        labels="position",
        samples_per_combination=50,
    )
    D_test = test_distribution.sample_das(
        tokenizer=tokenizer,
        base_patterns=[
            "ABB",
        ],
        source_patterns=["BAB"],
        labels="position",
        samples_per_combination=50,
    ) + test_distribution.sample_das(
        tokenizer=tokenizer,
        base_patterns=[
            "BAB",
        ],
        source_patterns=["ABB"],
        labels="position",
        samples_per_combination=50,
    )

    data = []

    batch_size = 20
    eval_every = 5
    initial_lr = 0.01
    n_epochs = 10
    aligning_stream = stream

    for aligning_pos in positions:
        for aligning_layer in layers:
            if debug:
                print(
                    f"finding name position at: pos->{aligning_pos}, "
                    f"layers->{aligning_layer}, stream->{stream}"
                )
            config, num_restores = path_patching_config(
                aligning_layer, 
                gpt2.config.n_layer, 
                low_rank_dimension,
                component=aligning_stream,
                unit="pos"
            )
            intervenable = IntervenableModel(config, gpt2)
            intervenable.set_device("cuda")
            intervenable.disable_model_gradients()
            total_step = 0
            optimizer = torch.optim.Adam(
                intervenable.get_trainable_parameters(), lr=initial_lr
            )
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, end_factor=0.1, total_iters=n_epochs
            )

            for epoch in range(n_epochs):
                torch.cuda.empty_cache()
                for batch_dataset in D_train.batches(batch_size=batch_size):
                    # prepare base
                    base_inputs = batch_dataset.base.tokens
                    b_s = base_inputs["input_ids"].shape[0]
                    for k, v in base_inputs.items():
                        if v is not None and isinstance(v, torch.Tensor):
                            base_inputs[k] = v.to(gpt2.device)
                    # prepare source
                    source_inputs = batch_dataset.source.tokens
                    for k, v in source_inputs.items():
                        if v is not None and isinstance(v, torch.Tensor):
                            source_inputs[k] = v.to(gpt2.device)
                    # prepare label
                    labels = batch_dataset.patched_answer_tokens[:, 0].to(
                        gpt2.device
                    )

                    assert all(x == 18 for x in batch_dataset.base.lengths)
                    assert all(x == 18 for x in batch_dataset.source.lengths)

                    _, counterfactual_outputs = intervenable(
                        {"input_ids": base_inputs["input_ids"]},
                        [{"input_ids": source_inputs["input_ids"]}, {"input_ids": base_inputs["input_ids"]}],
                        {
                            "sources->base": (
                                [[[aligning_pos]] * b_s]+[[[aligning_pos]] * b_s]*num_restores, 
                                [[[aligning_pos]] * b_s]+[[[aligning_pos]] * b_s]*num_restores, 
                            )
                        },
                    )

                    eval_metrics = compute_metrics(
                        [counterfactual_outputs.logits], [labels]
                    )
                    loss = calculate_loss(counterfactual_outputs.logits, labels)
                    loss_str = round(loss.item(), 2)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    intervenable.set_zero_grad()
                    total_step += 1

            # eval
            eval_labels = []
            eval_preds = []
            with torch.no_grad():
                torch.cuda.empty_cache()
                for batch_dataset in D_test.batches(batch_size=batch_size):
                    # prepare base
                    base_inputs = batch_dataset.base.tokens
                    b_s = base_inputs["input_ids"].shape[0]
                    for k, v in base_inputs.items():
                        if v is not None and isinstance(v, torch.Tensor):
                            base_inputs[k] = v.to(gpt2.device)
                    # prepare source
                    source_inputs = batch_dataset.source.tokens
                    for k, v in source_inputs.items():
                        if v is not None and isinstance(v, torch.Tensor):
                            source_inputs[k] = v.to(gpt2.device)
                    # prepare label
                    labels = batch_dataset.patched_answer_tokens[:, 0].to(gpt2.device)

                    assert all(x == 18 for x in batch_dataset.base.lengths)
                    assert all(x == 18 for x in batch_dataset.source.lengths)
                    _, counterfactual_outputs = intervenable(
                        {"input_ids": base_inputs["input_ids"]},
                        [{"input_ids": source_inputs["input_ids"]}, {"input_ids": base_inputs["input_ids"]}],
                        {
                            "sources->base": (
                                [[[aligning_pos]] * b_s]+[[[aligning_pos]] * b_s]*num_restores, 
                                [[[aligning_pos]] * b_s]+[[[aligning_pos]] * b_s]*num_restores, 
                            )
                        },
                    )
                    eval_labels += [labels]
                    eval_preds += [counterfactual_outputs.logits]
            eval_metrics = compute_metrics(eval_preds, eval_labels)

            data.append(
                {
                    "pos": aligning_pos,
                    "layer": aligning_layer,
                    "acc": eval_metrics["accuracy"],
                    "kl_div": eval_metrics["kl_div"],
                    "stream": stream,
                }
            )

    return data