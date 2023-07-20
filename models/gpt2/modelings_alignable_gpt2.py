from transformers import GPT2LMHeadModel, GPT2Model, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import copy

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from dataclasses import dataclass

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
class AlignableBaseModelOutputWithPastAndCrossAttentions(BaseModelOutputWithPastAndCrossAttentions):
    rotated_hidden_states: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class AlignableCausalLMOutputWithCrossAttentions(CausalLMOutputWithCrossAttentions):
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

class AlignableGPT2Model(GPT2Model):
    def __init__(self, config, alignment_config=None):
        super().__init__(config)
        self.wte.requires_grad_(False)
        self.alignment_config = alignment_config
        
        if self.alignment_config != None:
            self.n_embd = config.n_embd
            # create rotate and derotate layers for alignment
            searchable_n_embd = (
                int(alignment_config["num_of_das_token"])
            ) * config.n_embd
            rotate_layer = RotateLayer(searchable_n_embd)
            self.searchable_n_embd = searchable_n_embd
            self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
            self.inverse_rotate_layer = InverseRotateLayer(self.rotate_layer)
        
            # this will be replaced by a learnable parameters
            # TODO: now, we can only align two variable at the same time.
            self.intervention_boundaries = torch.tensor([1.0, 1.0], requires_grad=True)
            self.intervention_boundaries = torch.nn.Parameter(self.intervention_boundaries)
            self.temperature = nn.Parameter(torch.tensor(50.0)) # Define temperature as a parameter
            self.intervention_population = nn.Parameter(torch.arange(0, self.searchable_n_embd), requires_grad=False)
            
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
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
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
            # TODO: potentially adding some noises to non-special tokens.
            
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            hidden_states = outputs[0]
            if self.alignment_config != None and i == self.alignment_config["layer"]:
                # disjoin
                original_shape = copy.deepcopy(hidden_states.shape)
                hidden_states = hidden_states.reshape(batch_size, -1)
                aligning_hidden_states = []
                for i in range(batch_size):
                    if intervention_token_range is not None:
                        start = intervention_token_range[i][0]*self.n_embd
                        end = intervention_token_range[i][1]*self.n_embd
                    else:
                        start = self.alignment_config["token_range"][0]*self.hidden_size
                        end = self.alignment_config["token_range"][1]*self.hidden_size
                    aligning_hidden_state = hidden_states[i,start:end]
                    aligning_hidden_states += [aligning_hidden_state]
                aligning_hidden_states = torch.stack(aligning_hidden_states, dim=0)
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
                # rotate back
                reversed_hidden_states = self.inverse_rotate_layer(rotated_hidden_states)
                # suture back iteratively
                intervened_hidden_states = []
                for i in range(batch_size):
                    if intervention_token_range is not None:
                        start = intervention_token_range[i][0]*self.n_embd
                        end = intervention_token_range[i][1]*self.n_embd
                    else:
                        start = self.alignment_config["token_range"][0]*self.hidden_size
                        end = self.alignment_config["token_range"][1]*self.hidden_size
                    prefix_hidden_state = hidden_states[i,:start]
                    postfix_hidden_state = hidden_states[i,end:]
                    intervened_hidden_state = torch.cat([
                        prefix_hidden_state,
                        reversed_hidden_states[i],
                        postfix_hidden_state
                    ], dim=-1)
                    intervened_hidden_states += intervened_hidden_state
                hidden_states = torch.stack(intervened_hidden_states, dim=0)
                hidden_states = hidden_states.reshape(original_shape)
                if output_rotated_hidden_states_only:
                    # we early exist.
                    return AlignableBaseModelOutputWithPastAndCrossAttentions(
                        last_hidden_state=hidden_states,
                        past_key_values=presents,
                        hidden_states=all_hidden_states,
                        attentions=all_self_attentions,
                        cross_attentions=all_cross_attentions,
                        rotated_hidden_states=rotated_hidden_states
                    )

            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions, rotated_hidden_states]
                if v is not None
            )

        return AlignableBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
            rotated_hidden_states=rotated_hidden_states
        )

class AlignableGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config, alignment_config):
        super().__init__(config)
        self.transformer = AlignableGPT2Model(config, alignment_config)
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
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
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            ########################################
            # sources related information goes here
            ########################################
            source_hidden_states=source_hidden_states,
            intervention_ids=intervention_ids,
            intervention_token_range=intervention_token_range,
            output_rotated_hidden_states_only=output_rotated_hidden_states_only,
            ########################################
            # sources related information ends here
            ########################################
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        rotated_hidden_states = transformer_outputs[-1]
        
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # boundary loss - make it smaller
            if self.transformer.alignment_config != None:
                boundary_loss = 1. * self.transformer.intervention_boundaries.sum()
                loss += boundary_loss
                
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return AlignableCausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
            rotated_hidden_states=rotated_hidden_states
        )
    