from collections import OrderedDict
from typing import Any, List, Mapping, Optional

from transformers import PreTrainedTokenizer, TensorType, is_torch_available
from transformers.configuration_utils import PretrainedConfig


class AlignableLlamaConfig(PretrainedConfig):
    model_type = "llama"

    def __init__(self, das_layer=15, das_token_range=[80, 81], **kwargs):
        self.das_layer = das_layer
        self.das_token_range = das_token_range

        super().__init__(**kwargs)


class AlignableT5Config(PretrainedConfig):
    model_type = "t5"

    def __init__(self,
                 das_layer=15,
                 das_token_range=[80, 81],
                 alignment_stack='decoder',
                 **kwargs):
        self.das_layer = das_layer
        self.das_token_range = das_token_range
        self.alignment_stack = alignment_stack

        super().__init__(**kwargs)
