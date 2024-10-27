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

vlbart_multitask_type_to_module_mapping = {
    "block_input": ("encoder.layers[%s]", CONST_INPUT_HOOK),
    "block_output": ("encoder.layers[%s]", CONST_OUTPUT_HOOK),
    "decoder_block_input": ("decoder.layers[%s]", CONST_INPUT_HOOK),
    "decoder_block_output": ("decoder.layers[%s]", CONST_OUTPUT_HOOK),
    "mlp_activation": ("encoder.layers[%s].fc1.act_fn", CONST_OUTPUT_HOOK),
    "mlp_output": ("encoder.layers[%s].fc2", CONST_OUTPUT_HOOK),
    "mlp_input": ("encoder.layers[%s].fc1", CONST_INPUT_HOOK),
    "attention_value_output": ("encoder.layers[%s].self_attn.out_proj", CONST_INPUT_HOOK),
    "head_attention_value_output": ("encoder.layers[%s].self_attn.out_proj", CONST_INPUT_HOOK),
    "attention_output": ("encoder.layers[%s].self_attn", CONST_OUTPUT_HOOK),
    "attention_input": ("encoder.layers[%s].self_attn", CONST_INPUT_HOOK),
    "query_output": ("encoder.layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK),
    "key_output": ("encoder.layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK),
    "value_output": ("encoder.layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK),
    "head_query_output": ("encoder.layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK),
    "head_key_output": ("encoder.layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK),
    "head_value_output": ("encoder.layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK),
}


vlbart_multitask_type_to_dimension_mapping = {
    "block_input": ("d_model",),
    "block_output": ("d_model",),
    "decoder_block_input": ("d_model",),
    "decoder_block_output": ("d_model",),
    "mlp_activation": ("encoder_ffn_dim",),
    "mlp_output": ("d_model",),
    "mlp_input": ("d_model",),
    "attention_value_output": ("d_model",),
    "head_attention_value_output": ("d_model/encoder_attention_heads",),
    "attention_output": ("d_model",),
    "attention_input": ("d_model",),
    "query_output": ("d_model",),
    "key_output": ("d_model",),
    "value_output": ("d_model",),
    "head_query_output": ("d_model/encoder_attention_heads",),
    "head_key_output": ("d_model/encoder_attention_heads",),
    "head_value_output": ("d_model/encoder_attention_heads",),
}


"""llava model with LM head"""
vlbart_multitask_lm_type_to_module_mapping = {}
for k, v in vlbart_multitask_type_to_module_mapping.items():
    vlbart_multitask_lm_type_to_module_mapping[k] = (f"model.{v[0]}", v[1])


vlbart_multitask_lm_type_to_dimension_mapping = vlbart_multitask_type_to_dimension_mapping
