from collections.abc import Callable
import collections
import os
from pathlib import Path
import random

from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from .task_base import TaskBase


class ContinentMatchingTask(TaskBase):

    def __init__(self):
        self.country_to_continent = self._get_country_continent_map()
        self.continent_to_country = collections.defaultdict(list)
        for country, continent in self.country_to_continent.items():
            self.continent_to_country[continent].append(country)
        self.countries = [c for c in self.country_to_continent]
        self.continents = [c for c in self.continent_to_country]

    def _get_country_continent_map(self):
        ret = {}
        base_data = os.path.join(
            Path(__file__).parent, 'countries_continents.csv')
        with open(base_data) as f:
            for line in f:
                line = line.strip()
                comma_parts = line.split(',')
                continent = comma_parts[0]
                if continent not in ('Asia', 'Africa', 'North America',
                                     'South America', 'Oceania', 'Europe'):
                    continue
                country = ','.join(comma_parts[1:])
                ret[country] = continent
            return ret

    def _format_prompt(self,
                       country1: str,
                       country2: str,
                       custom_prompt_fn: Callable = None):
        if custom_prompt_fn:
            return custom_prompt_fn(country1, country2)
        return f'''Answer the following yes/no question:\n\nAre {country1} and {country2} on the same continent?\n'''

    def _sample_single_example(self,
                               target_label: bool = None,
                               custom_prompt_fn: Callable = None):
        '''
        Sample a single example.

        Args:
            target_label: if set (to either True or False), will sample an example that has this label. Otherwise will sample randomly.
            custom_prompt_fn: if set, this must be a function that accepts two countries and formats a (string) input prompt.

        Returns: a dictionary with keys 'inputs' and 'targets', both with values encoded as strings.
        '''
        # First sample country1
        country1 = random.choice(self.countries)
        continent1 = self.country_to_continent[country1]

        if target_label is None:
            # sample randomly
            country2 = random.choice(self.countries)
            continent2 = self.country_to_continent[country2]
        elif target_label:
            # sample randomly from the same continent
            country2 = random.choice(self.continent_to_country[continent1])
            continent2 = self.country_to_continent[country2]
        else:
            # sample randomly from the other continents
            continent2 = random.choice(
                [c for c in self.continents if c != continent1])
            country2 = random.choice(self.continent_to_country[continent2])

        example = {
            'inputs': self._format_prompt(country1, country2,
                                          custom_prompt_fn),
            'targets': 'Yes' if continent1 == continent2 else 'No'
        }
        print('sampled', example)
        return example

    def _encode_examples(self, tokenizer, examples: list[dict[str, str]]):
        '''
        Returns input_ids, target_ids (both lists of lists of ints)
        '''
        all_input_ids = []
        all_output_ids = []
        for example in examples:
            input_ids = tokenizer(example['inputs'],
                                  return_tensors='pt',
                                  padding='max_length',
                                  max_length=40).input_ids[0]
            label = tokenizer.convert_tokens_to_ids(example['targets'])
            output_ids = (torch.ones(input_ids.shape[0]) *
                          -100).long().tolist()
            output_ids[-1] = label
            input_ids = input_ids.tolist()
            all_input_ids.append(input_ids)
            all_output_ids.append(output_ids)
        return all_input_ids, all_output_ids

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

        prealign_str_examples = []
        for _ in range(n_eval // 2):
            prealign_str_examples.append(self._sample_single_example(True))
            prealign_str_examples.append(self._sample_single_example(False))

        # Encode the prealign examples
        input_ids, output_ids = self._encode_examples(tokenizer,
                                                      prealign_str_examples)

        prealign_dataset = Dataset.from_dict({
            "input_ids": input_ids,
            "labels": output_ids,
        }).with_format("torch")
        prealign_dataloader = DataLoader(prealign_dataset,
                                         batch_size=prealign_batch_size)

        return prealign_dataloader, prealign_dataloader, prealign_dataloader, prealign_dataloader
