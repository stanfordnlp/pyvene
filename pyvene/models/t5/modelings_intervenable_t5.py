"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.

We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.
"""


from ..constants import CONST_INPUT_HOOK, CONST_OUTPUT_HOOK


t5_lm_type_to_module_mapping = {
    "mlp_output": ("encoder.block[%s].layer[1]", CONST_OUTPUT_HOOK),
    "attention_input": ("encoder.block[%s].layer[0]", CONST_OUTPUT_HOOK),
    "encoder.block_input": ("encoder.block[%s]", CONST_INPUT_HOOK),
    "encoder.block_output": ("encoder.block[%s]", CONST_OUTPUT_HOOK),
    "encoder.mlp_activation": ("encoder.block[%s].layer[1].DenseReluDense.wo", CONST_OUTPUT_HOOK),
    "encoder.mlp_output": ("encoder.block[%s].layer[1]", CONST_OUTPUT_HOOK),
    "encoder.mlp_input": ("encoder.block[%s].layer[1]", CONST_INPUT_HOOK),
    "encoder.attention_value_output": ("encoder.block[%s].layer[0].SelfAttention.o", CONST_INPUT_HOOK),
    "encoder.head_attention_value_output": ("encoder.block[%s].layer[0].SelfAttention.o", CONST_INPUT_HOOK),
    "encoder.attention_output": ("encoder.block[%s].layer[0]", CONST_OUTPUT_HOOK),
    "encoder.attention_input": ("encoder.block[%s].layer[0]", CONST_INPUT_HOOK),
    "query_output": ("encoder.block[%s].layer[0].SelfAttention.q", CONST_OUTPUT_HOOK),
    "key_output": ("encoder.block[%s].layer[0].SelfAttention.k", CONST_OUTPUT_HOOK),
    "value_output": ("encoder.block[%s].layer[0].SelfAttention.v", CONST_OUTPUT_HOOK),
    "head_query_output": ("encoder.block[%s].layer[0].SelfAttention.q", CONST_OUTPUT_HOOK),
    "head_key_output": ("encoder.block[%s].layer[0].SelfAttention.k", CONST_OUTPUT_HOOK),
    "head_value_output": ("encoder.block[%s].layer[0].SelfAttention.v", CONST_OUTPUT_HOOK),
}


t5_lm_type_to_dimension_mapping = {
    "block_input": ("d_model",),
    "block_output": ("d_model",),
    "mlp_activation": ("d_ff",),
    "mlp_output": ("d_model",),
    "mlp_input": ("d_model",),
    "attention_value_output": ("d_model",),
    "head_attention_value_output": ("d_model/num_heads",),
    "attention_output": ("d_model",),
    "attention_input": ("d_model",),
    "query_output": ("d_model",),
    "key_output": ("d_model",),
    "value_output": ("d_model",),
    "head_query_output": ("d_model/num_heads",),
    "head_key_output": ("d_model/num_heads",),
    "head_value_output": ("d_model/num_heads",),
}