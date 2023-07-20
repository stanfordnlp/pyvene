from transformers import LlamaForCausalLM, LlamaModel, AutoConfig, AutoTokenizer, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from typing import Optional, Tuple, Union, List
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import copy
from dataclasses import dataclass
from transformers import get_linear_schedule_with_warmup
import numpy as np
import os, random, argparse, sys, pickle, time
from datasets import Dataset 
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import os, random, argparse, sys, pickle, time

from transformers.modeling_outputs import (
    BaseModelOutputWithPast, 
    CausalLMOutputWithPast
)

from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger("transformers")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sigmoid_boundary_sigmoid(_input, boundary_x, boundary_y, temperature):
    return torch.sigmoid((_input - boundary_x) / temperature) * \
        torch.sigmoid((boundary_y - _input) / temperature)

def harmonic_boundary_sigmoid(_input, boundary_x, boundary_y, temperature):
    return (_input<=boundary_x)*torch.sigmoid((_input - boundary_x) / temperature) + \
    (_input>=boundary_y)*torch.sigmoid((boundary_y - _input) / temperature) + \
    ((_input>boundary_x)&(_input<boundary_y))*torch.sigmoid(
        (0.5 * (torch.abs(_input - boundary_x)**(-1) + torch.abs(_input - boundary_y)**(-1)))**(-1) / temperature
    )
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@dataclass
class AlignableBaseModelOutputWithPast(BaseModelOutputWithPast):
    rotated_hidden_states: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class AlignableCausalLMOutputWithPast(CausalLMOutputWithPast):
    rotated_hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class InverseRotateLayer(torch.nn.Module):
    """The inverse of a given `LinearLayer` module."""
    def __init__(self, lin_layer):
        super().__init__()
        self.lin_layer = lin_layer

    def forward(self, x):
        output = torch.matmul(x, self.lin_layer.weight.T)
        return output

class RotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""
    def __init__(self, n, init_orth=True):
        super().__init__()
        weight = torch.empty(n,n)
        # we don't need init if the saved checkpoint has a nice 
        # starting point already.
        # you can also study this if you want, but it is our focus.
        if init_orth:
            torch.nn.init.orthogonal_(weight)
        self.weight = torch.nn.Parameter(weight, requires_grad=True)
        
    def forward(self, x):
        return torch.matmul(x, self.weight)
    
class AlignableLlamaModel(LlamaModel):
    def __init__(self, config, alignment_config=None):
        super().__init__(config)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        
        # this alignment config is like the whole population of neurons
        # you will be aligning with.
        self.alignment_config = alignment_config
        if self.alignment_config != None:
            self.hidden_size = config.hidden_size
            # create rotate and derotate layers for alignment
            searchable_n_embd = (
                alignment_config["token_range"][1] - alignment_config["token_range"][0]
            ) * config.hidden_size
            self.searchable_n_embd = searchable_n_embd
            rotate_layer = RotateLayer(searchable_n_embd)
            self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer, use_trivialization=False)
            self.inverse_rotate_layer = InverseRotateLayer(self.rotate_layer)

            # this will be replaced by a learnable parameters
            self.intervention_boundaries = torch.tensor([1.0, 1.0], requires_grad=True)
            self.intervention_boundaries = torch.nn.Parameter(self.intervention_boundaries)
            self.temperature = nn.Parameter(torch.tensor(50.0)) # Define temperature as a parameter
            self.intervention_population = nn.Parameter(torch.arange(0, self.searchable_n_embd), requires_grad=False)
            
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        ########################################
        # sources related information goes here
        ########################################
        source_hidden_states=None,
        intervention_ids=None,
        intervention_token_range=None,
        output_rotated_hidden_states_only: Optional[bool] = False,
        ########################################
        # sources related information ends here
        ########################################
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        rotated_hidden_states = None
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            
            if self.alignment_config != None and idx == self.alignment_config["layer"]:
                # disjoin
                original_shape = copy.deepcopy(hidden_states.shape)
                hidden_states = hidden_states.reshape(batch_size, -1)
                start = self.alignment_config["token_range"][0]*self.hidden_size
                end = self.alignment_config["token_range"][1]*self.hidden_size
                aligning_hidden_states = hidden_states[:,start:end]
                prefix_hidden_states = hidden_states[:,:start]
                postfix_hidden_states = hidden_states[:,end:]
                rotated_hidden_states = self.rotate_layer(aligning_hidden_states)
                                
                # intervene
                if source_hidden_states != None:
                    # boundary learning
                    intervention_boundaries = torch.clamp(self.intervention_boundaries, 1e-3, 1)
                    intervention_boundaries = torch.cumsum(intervention_boundaries, dim=0)
                    first_boundary_mask = sigmoid_boundary_sigmoid(
                        self.intervention_population.repeat(batch_size, 1), 
                        0.,
                        intervention_boundaries[0] * int(self.searchable_n_embd//2),
                        self.temperature
                    )
                    second_boundary_mask = sigmoid_boundary_sigmoid(
                        self.intervention_population.repeat(batch_size, 1), 
                        intervention_boundaries[0] * int(self.searchable_n_embd//2),
                        2 * intervention_boundaries[0] * int(self.searchable_n_embd//2),
                        self.temperature
                    )
                    boundary_mask = (intervention_ids==0).unsqueeze(dim=-1)*first_boundary_mask + \
                        (intervention_ids==1).unsqueeze(dim=-1)*second_boundary_mask
                    boundary_mask = boundary_mask.to(rotated_hidden_states.dtype)
                    
                    rotated_hidden_states = (1. - boundary_mask)*rotated_hidden_states + \
                        boundary_mask*source_hidden_states
                    
                # rotate back + suture
                reversed_hidden_states = self.inverse_rotate_layer(rotated_hidden_states)
                
                hidden_states = torch.cat([
                    prefix_hidden_states,
                    reversed_hidden_states,
                    postfix_hidden_states
                ], dim=1).reshape(original_shape)
                if output_rotated_hidden_states_only:
                    # we early exist.
                    return AlignableBaseModelOutputWithPast(
                        last_hidden_state=hidden_states,
                        past_key_values=next_decoder_cache if use_cache else None,
                        hidden_states=all_hidden_states,
                        attentions=all_self_attns,
                        rotated_hidden_states=rotated_hidden_states,
                    )
                
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, rotated_hidden_states] if v is not None)
        return AlignableBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            rotated_hidden_states=rotated_hidden_states,
        )
        
        
class AlignableLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, alignment_config=None):
        super().__init__(config)
        self.model = AlignableLlamaModel(config, alignment_config)
        if alignment_config != None:
            searchable_n_embd = (
                alignment_config["token_range"][1] - alignment_config["token_range"][0]
            ) * config.hidden_size
            self.searchable_n_embd = searchable_n_embd
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        ########################################
        # sources related information goes here
        ########################################
        source_input_ids=None,
        source_hidden_states=None,
        intervention_ids=None,
        intervention_token_range=None,
        output_rotated_hidden_states_only: Optional[bool] = False,
        ########################################
        # sources related information ends here
        ########################################
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        Example:
        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM
        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            ########################################
            # sources related information goes here
            ########################################
            source_hidden_states=source_hidden_states,
            intervention_ids=intervention_ids,
            output_rotated_hidden_states_only=output_rotated_hidden_states_only,
            ########################################
            # sources related information ends here
            ########################################
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        rotated_hidden_states = outputs[-1]
        logits = None
        if not output_rotated_hidden_states_only:
            logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :, :].contiguous()
            shift_labels = labels[..., :].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            
            # boundary loss - make it smaller
            if self.model.alignment_config != None:
                boundary_loss = 1. * self.model.intervention_boundaries.sum()
                loss += boundary_loss
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return AlignableCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rotated_hidden_states=rotated_hidden_states
        )