import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.pytorch_utils import Conv1D
from transformers.utils import (
    ModelOutput,
    logging,
)
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2PreTrainedModel

logger = logging.get_logger(__name__)


class BackpackGPT2Config(GPT2Config):
  """
    This is the configuration class to store the configuration of a [`GPT2Model`] or a [`TFGPT2Model`]. It is used to
    instantiate a Backpack GPT-2 model according to the specified arguments, defining the model architecture.
    Configuration objects inherit from [`GPT2Config`] and can be used to control the model outputs. Read the
    documentation from [`GPT2Config`] for more information.
    Args:
        num_senses (`int`, *optional*, defaults to 16):
            The number of sense vectors to define for each word.
        sense_intermediate_scale (`int`, *optional*, defaults ot 4):
            The hidden dimensionality of the sense vector network.
    Example:
    ```python
    >>> from transformers import BackpackGPT2Config, BackpackGPT2Model
    >>> # Initializing a GPT2 configuration
    >>> configuration = BackpackGPT2Config()
    >>> # Initializing a model (with random weights) from the configuration
    >>> model = BackpackGPT2Model(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
  """

  def __init__(self,
               vocab_size=50264,
               num_senses=16,
               sense_intermediate_scale=4,
               n_positions=512,
               scale_attn_by_inverse_layer_idx=True,
               **kwargs,
  ):
    self.num_senses = num_senses
    self.sense_intermediate_scale = sense_intermediate_scale
    super().__init__(vocab_size=vocab_size, n_positions=n_positions,
                     scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx, **kwargs)
    

### Backpack-Specific
class BackpackGPT2PreTrainedModel(GPT2PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias"]

    config_class = BackpackGPT2Config
    base_model_prefix = "backpack"
    is_parallelizable = True
    supports_gradient_checkpointing = False
    _no_split_modules = ["GPT2Block", "BackpackNoMixBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

class BackpackMLP(nn.Module):

    def __init__(self, embed_dim, intermediate_dim, out_dim, config):
        super().__init__()
        self.c_fc = Conv1D(intermediate_dim, embed_dim)
        self.c_proj = Conv1D(out_dim, intermediate_dim)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class BackpackNoMixBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = BackpackMLP(config.n_embd, config.n_embd*4, config.n_embd, config)
        self.resid_dropout1 = nn.Dropout(config.resid_pdrop)
        self.resid_dropout2 = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states, residual):
        residual = self.resid_dropout1(hidden_states) + residual
        hidden_states = self.ln_1(residual)
        mlp_out = self.mlp(hidden_states)
        residual = self.resid_dropout2(mlp_out) + residual
        hidden_states = self.ln_2(residual)
        return hidden_states


class BackpackSenseNetwork(nn.Module):
    def __init__(self, config, num_senses, device=None, dtype=None):
        super().__init__()
        self.num_senses = num_senses
        #self.embeddings = embeddings
        self.n_embd = config.n_embd

        self.dropout = nn.Dropout(config.embd_pdrop)
        self.block = BackpackNoMixBlock(config)
        self.ln = nn.LayerNorm(self.n_embd, eps=config.layer_norm_epsilon)
        self.final_mlp = BackpackMLP(
            embed_dim=config.n_embd,
            intermediate_dim=config.sense_intermediate_scale*config.n_embd,
            out_dim=config.n_embd*config.num_senses,
            config=config,
            )

    def forward(self, input_embeds):
        residual = self.dropout(input_embeds)
        hidden_states = self.ln(residual)
        hidden_states = self.block(hidden_states, residual)
        senses = self.final_mlp(hidden_states)
        bs, s, nvd = senses.shape
        return senses.reshape(bs, s, self.num_senses, self.n_embd).transpose(1,2) # (bs, nv, s, d)

class BackpackWeightNetwork(nn.Module):

    def __init__(self, num_senses, embed_dim):
        super().__init__()
        self.n_embd = embed_dim
        self.num_senses = num_senses
        self.embed_per_sense = embed_dim // num_senses
        self.c_attn = nn.Linear(embed_dim, 2 * num_senses * self.embed_per_sense)
        self.softmax_scale = None

    def forward(self, encoded):
        b, s, d = encoded.shape
        encoded = self.c_attn(encoded) # (b, s, 2*d)
        encoded = encoded.reshape(b, s, 2, self.num_senses, self.embed_per_sense) #(b, s, 2, nv, d//nv)
        batch_size, seqlen = encoded.shape[0], encoded.shape[1]

        # compute scores & mask
        q, k = encoded.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum('bthd,bshd->bhts', q, k * softmax_scale)
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        scores = scores + causal_mask.to(dtype=scores.dtype)

        return torch.softmax(scores, dim=-1, dtype=q.dtype)
    

@dataclass
class BackpackGPT2BaseModelOutput(ModelOutput):
    hidden_states: torch.FloatTensor = None
    contextualization: torch.FloatTensor = None

class BackpackGPT2Model(BackpackGPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r".*attn.masked_bias", r".*attn.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.n_embd

        self.num_senses = config.num_senses
        self.gpt2_model = GPT2Model(config)
        self.sense_network = BackpackSenseNetwork(config, self.num_senses, self.gpt2_model.wte)
        self.word_embeddings = self.gpt2_model.wte
        self.position_embeddings = self.gpt2_model.wpe
        self.sense_weight_net = BackpackWeightNetwork(self.num_senses, self.embed_dim)
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def get_num_senses(self):
        return self.num_senses

    def get_word_embeddings(self):
        return self.word_embeddings

    def get_sense_network(self):
        return self.sense_network

    def forward(self, input_ids, position_ids):
        # Compute senses
        sense_input_embeds = self.word_embeddings(input_ids)
        senses = self.sense_network(sense_input_embeds) # (bs, nv, s, d)

        # Compute contextualization weights
        contextl_hidden_states = self.gpt2_model(input_ids, position_ids=position_ids).last_hidden_state # (bs, s, d)
        contextualization = self.sense_weight_net(contextl_hidden_states) # (bs, nv, s, s)

        # Compute resulting outputs
        hidden_states = torch.sum(contextualization @ senses, dim=1) # (bs, nv, s, d) -> (bs, s, d)
        return BackpackGPT2BaseModelOutput(
            hidden_states=hidden_states,
            contextualization=contextualization,
        )
    
    def run_with_custom_contextualization(self, input_ids, contextualization):
        # Compute senses
        sense_input_embeds = self.word_embeddings(input_ids)
        senses = self.sense_network(sense_input_embeds) # (bs, nv, s, d)

        # Compute resulting outputs
        hidden_states = torch.sum(contextualization @ senses, dim=1) # (bs, nv, s, d) -> (bs, s, d)
        return BackpackGPT2BaseModelOutput(
            hidden_states=hidden_states,
            contextualization=contextualization,
        )

@dataclass
class BackpackGPT2LMHeadModelOutput(ModelOutput):
    logits: torch.FloatTensor = None
    contextualization: torch.FloatTensor = None

class BackpackGPT2LMHeadModel(BackpackGPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r".*attn.masked_bias", r".*attn.bias"]

    def __init__(self, config):
        super().__init__(config)
        self.backpack = BackpackGPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.backpack.word_embeddings.weight # also tied with the underlying underlying transf

    def get_lm_head(self):
        return self.lm_head

    def forward(self, input_ids, position_ids=None):
        outputs = self.backpack(input_ids, position_ids=position_ids)
        hidden_states, contextualization = outputs.hidden_states, outputs.contextualization
        lm_logits = self.lm_head(hidden_states) # (bs, s, V)
        return BackpackGPT2LMHeadModelOutput(
            logits=lm_logits,
            contextualization=contextualization,
        )

    def run_with_custom_contextualization(self, input_ids, contextualization):
        outputs = self.backpack.run_with_custom_contextualization(input_ids, contextualization)
        hidden_states, contextualization = outputs.hidden_states, outputs.contextualization
        lm_logits = self.lm_head(hidden_states)
        return BackpackGPT2LMHeadModelOutput(
        logits=lm_logits,
        contextualization=contextualization,
    )
            