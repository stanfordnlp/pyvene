import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import Optional, Tuple
import numpy as np
from dataclasses import dataclass
from transformers.utils import ModelOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    ModelOutput,
    SequenceClassifierOutput,
    CausalLMOutput,
)


class GRUConfig(PretrainedConfig):
    model_type = "gru"

    def __init__(
        self,
        include_emb=False,
        vocab_size=50_257,
        max_position_embeddings=512,
        n_layer=2,
        h_dim=512,
        n_labels=2,
        include_bias=True,
        pdrop=0.3,
        problem_type="single_label_classification",
        initializer_range=0.02,
        **kwargs,
    ):
        self.include_emb = include_emb
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.n_layer = n_layer
        self.h_dim = h_dim
        self.include_bias = include_bias
        self.pdrop = pdrop
        self.n_labels = n_labels
        self.problem_type = problem_type
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


class GRUCell(nn.Module):
    def __init__(self, config):
        super(GRUCell, self).__init__()
        self.h_dim = config.h_dim
        self.include_bias = config.include_bias

        self.x2h = nn.Linear(self.h_dim, 3 * self.h_dim, bias=self.include_bias)
        self.h2h = nn.Linear(self.h_dim, 3 * self.h_dim, bias=self.include_bias)

        self.reset_act = nn.Sigmoid()
        self.update_act = nn.Sigmoid()
        self.new_act = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.h_dim)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, current_states, hidden_states=None):
        if hidden_states is None:
            hidden_states = Variable(
                current_states.new_zeros(current_states.size(0), self.hidden_size)
            )

        x_t = self.x2h(current_states)
        h_t = self.h2h(hidden_states)

        x_reset, x_upd, x_new = x_t.chunk(3, 1)
        h_reset, h_upd, h_new = h_t.chunk(3, 1)

        reset_gate = self.reset_act(x_reset + h_reset)
        update_gate = self.update_act(x_upd + h_upd)
        new_gate = self.new_act(x_new + (reset_gate * h_new))

        hy = update_gate * hidden_states + (1 - update_gate) * new_gate

        return hy


@dataclass
class GRUModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class GRUPreTrainedModel(PreTrainedModel):
    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class GRUModel(GRUPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.h_dim = config.h_dim
        self.include_bias = config.include_bias
        self.n_layer = config.n_layer
        if config.include_emb:
            self.wte = nn.Embedding(config.vocab_size, self.h_dim)
            self.wpe = nn.Embedding(config.max_position_embeddings, self.h_dim)

        self.cells = nn.ModuleList(
            [GRUCell(self.config) for _ in range(0, self.n_layer)]
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        if position_ids is not None:
            position_embeds = self.wpe(position_ids)
            inputs_embeds += position_embeds

        batch_size = inputs_embeds.shape[0]
        max_seq_len = inputs_embeds.shape[1]
        if hidden_states is None:
            h0 = Variable(torch.zeros(self.n_layer, batch_size, self.h_dim)).to(
                inputs_embeds.device
            )
        else:
            h0 = hidden_states
        all_layer_hidden_states = [h0[layer, :, :] for layer in range(self.n_layer)]

        all_hidden_states = []
        for t in range(max_seq_len):
            for layer in range(self.n_layer):
                if layer == 0:
                    current_layer_hidden_state = self.cells[layer](
                        inputs_embeds[:, t, :], all_layer_hidden_states[layer]
                    )
                else:
                    current_layer_hidden_state = self.cells[layer](
                        all_layer_hidden_states[layer - 1],
                        all_layer_hidden_states[layer],
                    )
                all_layer_hidden_states[layer] = current_layer_hidden_state

            all_hidden_states.append(current_layer_hidden_state)

        all_hidden_states = torch.stack(all_hidden_states, dim=1)

        if not return_dict:
            return tuple(
                v
                for v in [all_hidden_states, current_layer_hidden_state]
                if v is not None
            )

        return GRUModelOutput(
            hidden_states=all_hidden_states,
            last_hidden_state=current_layer_hidden_state,
        )


class GRUForClassification(GRUPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.n_labels = config.n_labels
        self.gru = GRUModel(config)
        self.score = nn.Linear(config.h_dim, self.n_labels, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        gru_outputs = self.gru(
            input_ids,
            position_ids,
            inputs_embeds,
            output_hidden_states,
            return_dict,
        )
        hidden_states = gru_outputs[0]

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if attention_mask is None:
            if input_ids is not None:
                sequence_lengths = torch.ones_like(input_ids).sum(dim=-1).int() - 1
            else:
                sequence_lengths = (
                    torch.ones(inputs_embeds.shape[0], inputs_embeds.shape[1])
                    .to(inputs_embeds.device)
                    .sum(dim=-1)
                    .int()
                    - 1
                )
        else:
            sequence_lengths = attention_mask.sum(dim=-1).int() - 1

        pooled_hidden_states = hidden_states[
            torch.arange(batch_size, device=hidden_states.device), sequence_lengths
        ]
        pooled_logits = self.score(pooled_hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        if not return_dict:
            output = (pooled_logits,) + gru_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=gru_outputs.hidden_states,
        )


class GRULMHeadModel(GRUPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.n_labels = config.n_labels
        self.gru = GRUModel(config)
        self.lm_head = nn.Linear(config.h_dim, config.vocab_size, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        gru_outputs = self.gru(
            input_ids,
            position_ids,
            inputs_embeds,
            output_hidden_states,
            return_dict,
        )
        hidden_states = gru_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + gru_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=lm_logits,
            hidden_states=gru_outputs.hidden_states,
        )
