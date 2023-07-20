from collections import OrderedDict
from typing import Any, List, Mapping, Optional

from transformers import PreTrainedTokenizer, TensorType, is_torch_available
from transformers.configuration_utils import PretrainedConfig


class AlignableLlamaConfig(PretrainedConfig):
    model_type="llama"
    def __init__(
        self,
        das_layer=15,
        das_token_range=[80, 81],
        num_of_das_token=1,
        das_head=None,
        das_mode="residual", # mlp_residual, attention_block, head
        **kwargs
    ):
        self.das_layer = das_layer
        self.das_token_range = das_token_range
        self.num_of_das_token = num_of_das_token
        self.das_head = das_head
        self.das_mode = das_mode
        
        super().__init__(**kwargs)


class AlignableGPT2Config(PretrainedConfig):
    model_type="gpt2"
    def __init__(
        self,
        das_layer=6,
        das_token_range=None,
        num_of_das_token=1,
        das_head=None,
        das_mode="residual", # mlp_residual, attention_block, head
        **kwargs
    ):
        self.das_layer = das_layer
        self.das_token_range = das_token_range
        self.num_of_das_token = num_of_das_token
        self.das_head = das_head
        self.das_mode = das_mode
        
        super().__init__(**kwargs)