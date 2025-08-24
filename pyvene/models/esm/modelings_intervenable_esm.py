"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.

We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.
"""

from ..constants import (
    CONST_INPUT_HOOK,
    CONST_OUTPUT_HOOK
)

"""esm base model"""
esm_type_to_module_mapping = dict(
    block_input=("encoder.layer[%s]", CONST_INPUT_HOOK),
    block_output=("encoder.layer[%s]", CONST_OUTPUT_HOOK),

    mlp_input=("encoder.layer[%s].intermediate", CONST_INPUT_HOOK),
    mlp_activation=("encoder.layer[%s].intermediate", CONST_OUTPUT_HOOK),
    mlp_output=("encoder.layer[%s].output", CONST_OUTPUT_HOOK),

    attention_value_output=("encoder.layer[%s].attention.output", CONST_INPUT_HOOK),
    head_attention_value_output=("encoder.layer[%s].attention.output", CONST_INPUT_HOOK),

    attention_input=("encoder.layer[%s].attention", CONST_INPUT_HOOK),
    attention_output=("encoder.layer[%s].attention", CONST_OUTPUT_HOOK),

    query_output=("encoder.layer[%s].attention.self.query", CONST_OUTPUT_HOOK),
    head_query_output=("encoder.layer[%s].attention.self.query", CONST_OUTPUT_HOOK),
    key_output=("encoder.layer[%s].attention.self.key", CONST_OUTPUT_HOOK),
    head_key_output=("encoder.layer[%s].attention.self.key", CONST_OUTPUT_HOOK),
    value_output=("encoder.layer[%s].attention.self.value", CONST_OUTPUT_HOOK),
    head_value_output=("encoder.layer[%s].attention.self.value", CONST_OUTPUT_HOOK),

)
esm_type_to_dimension_mapping = dict(
    block_input=("hidden_size",),
    block_output=("hidden_size"),

    mlp_input=("hidden_size",),
    mlp_activation=("intermediate_size",),
    mlp_output=("hidden_size",),

    attention_value_output=("hidden_size",),
    head_attention_value_output=("hidden_size/num_attention_heads",),

    attention_input=("num_attention_heads",),
    attention_output=("num_attention_heads/num_attention_heads",),

    query_output=("num_attention_heads",),
    head_query_output=("num_attention_heads/num_attention_heads",),
    key_output=("num_attention_heads",),
    head_key_output=("num_attention_heads/num_attention_heads",),
    value_output=("num_attention_heads",),
    head_value_output=("num_attention_heads/num_attention_heads",),
)

"""esm for mlm model"""
esm_mlm_type_to_module_mapping = {k: ("esm." + i, j) for k, (i, j) in esm_type_to_module_mapping.items()}
esm_mlm_type_to_dimension_mapping = esm_type_to_dimension_mapping.copy()
