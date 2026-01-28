"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.
We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.
"""
import torch
from ..constants import *

w2v2bert_type_to_module_mapping = {
    "block_input": ("encoder.layers[%s]", CONST_INPUT_HOOK),
    "block_output": ("encoder.layers[%s]", CONST_OUTPUT_HOOK),
    "ffn1_activation": ("encoder.layers[%s].ffn1.intermediate_act_fn", CONST_OUTPUT_HOOK),
    "ffn1_output": ("encoder.layers[%s].ffn1", CONST_OUTPUT_HOOK),
    "ffn1_input": ("encoder.layers[%s].ffn1", CONST_INPUT_HOOK),
    "ffn2_activation": ("encoder.layers[%s].ffn2.intermediate_act_fn", CONST_OUTPUT_HOOK),
    "ffn2_output": ("encoder.layers[%s].ffn2", CONST_OUTPUT_HOOK),
    "ffn2_input": ("encoder.layers[%s].ffn2", CONST_INPUT_HOOK),
    "attention_value_output": ("encoder.layers[%s].self_attn.linear_out", CONST_INPUT_HOOK),
    "head_attention_value_output": ("encoder.layers[%s].self_attn.linear_out", CONST_INPUT_HOOK, (split_head_and_permute, "n_head")),
    "attention_output": ("encoder.layers[%s].self_attn", CONST_OUTPUT_HOOK),
    "attention_input": ("encoder.layers[%s].self_attn", CONST_INPUT_HOOK),
    "query_output": ("encoder.layers[%s].self_attn.linear_q", CONST_OUTPUT_HOOK),
    "key_output": ("encoder.layers[%s].self_attn.linear_k", CONST_OUTPUT_HOOK),
    "value_output": ("encoder.layers[%s].self_attn.linear_v", CONST_OUTPUT_HOOK),
    "head_query_output": ("encoder.layers[%s].self_attn.linear_q", CONST_OUTPUT_HOOK, (split_head_and_permute, "n_head")),
    "head_key_output": ("encoder.layers[%s].self_attn.linear_k", CONST_OUTPUT_HOOK, (split_head_and_permute, "n_head")),
    "head_value_output": ("encoder.layers[%s].self_attn.linear_v", CONST_OUTPUT_HOOK, (split_head_and_permute, "n_head")),
    "conv_output": ("encoder.layers[%s].conv_module", CONST_OUTPUT_HOOK),
    "conv_input": ("encoder.layers[%s].conv_module", CONST_INPUT_HOOK),
    "conv_glu_output": ("encoder.layers[%s].conv_module.glu", CONST_OUTPUT_HOOK),
    "conv_depth_output": ("encoder.layers[%s].conv_module.depthwise_conv", CONST_OUTPUT_HOOK),
}

w2v2bert_type_to_dimension_mapping = {
    "n_head": ("num_attention_heads",),
    "block_input": ("hidden_size",),
    "block_output": ("hidden_size",),
    "ffn1_activation": ("intermediate_size",),
    "ffn1_output": ("hidden_size",),
    "ffn1_input": ("hidden_size",),
    "ffn2_activation": ("intermediate_size",),
    "ffn2_output": ("hidden_size",),
    "ffn2_input": ("hidden_size",),
    "attention_value_output": ("hidden_size",),
    "head_attention_value_output": ("hidden_size/num_attention_heads",),
    "attention_output": ("hidden_size",),
    "attention_input": ("hidden_size",),
    "query_output": ("hidden_size",),
    "key_output": ("hidden_size",),
    "value_output": ("hidden_size",),
    "head_query_output": ("hidden_size/num_attention_heads",),
    "head_key_output": ("hidden_size/num_attention_heads",),
    "head_value_output": ("hidden_size/num_attention_heads",),
    "conv_output": ("hidden_size",),
    "conv_input": ("hidden_size",),
    "conv_glu_output": ("hidden_size",),
    "conv_depth_output": ("hidden_size",),
}
