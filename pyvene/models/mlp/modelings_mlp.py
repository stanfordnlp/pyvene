from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput
from transformers.modeling_outputs import SequenceClassifierOutput
from dataclasses import dataclass

class MLPConfig(PretrainedConfig):
    model_type = "mlp"

    def __init__(
        self,
        include_emb=False,
        vocab_size=50_257,
        max_position_embeddings=512,
        n_layer=2,
        h_dim=512,
        num_classes=2,
        activation_function="gelu",
        pdrop=0.3,
        problem_type="single_label_classification",
        include_bias=True,
        squeeze_output=True,
        **kwargs,
    ):
        self.include_emb = include_emb
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.n_layer = n_layer
        self.h_dim = h_dim
        self.activation_function = activation_function
        self.pdrop = pdrop
        self.num_classes = num_classes
        self.problem_type = problem_type
        self.include_bias = include_bias
        self.squeeze_output = squeeze_output
        super().__init__(**kwargs)

@dataclass
class MLPModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class MLPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ff1 = nn.Linear(config.h_dim, config.h_dim, bias=config.include_bias)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.pdrop)

    def forward(self, hidden_states):
        return self.dropout(self.act(self.ff1(hidden_states)))


class MLPModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.h_dim = config.h_dim
        if config.include_emb:
            self.wte = nn.Embedding(config.vocab_size, self.h_dim)
            self.wpe = nn.Embedding(config.max_position_embeddings, self.h_dim)
        self.dropout = nn.Dropout(config.pdrop)

        self.h = nn.ModuleList([MLPBlock(config) for _ in range(config.n_layer)])

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        hidden_states = inputs_embeds
        if position_ids is not None:
            position_embeds = self.wpe(position_ids)
            hidden_states += position_embeds

        hidden_states = self.dropout(hidden_states)
        all_hidden_states = () if output_hidden_states else None

        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = block(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return MLPModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states
        )


class MLPForClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_classes = config.num_classes
        self.squeeze_output = config.squeeze_output
        self.mlp = MLPModel(config)
        self.score = nn.Linear(config.h_dim, self.num_classes, bias=config.include_bias)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        mlp_outputs = self.mlp(
            input_ids,
            position_ids,
            inputs_embeds,
            output_hidden_states,
            return_dict,
        )
        hidden_states = mlp_outputs[0]
        pooled_logits = self.score(hidden_states)
        if self.squeeze_output:
            pooled_logits = pooled_logits.squeeze(1)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_classes == 1:
                    self.config.problem_type = "regression"
                elif self.num_classes > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_classes == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_classes), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        if not return_dict:
            output = (pooled_logits,) + mlp_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=mlp_outputs.hidden_states,
        )
