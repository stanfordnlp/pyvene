"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.

We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.
"""


from ..constants import CONST_INPUT_HOOK, CONST_OUTPUT_HOOK, CONST_QKV_INDICES


phi_lm_type_to_module_mapping = {
    "block_input": ("layers[%s]", CONST_INPUT_HOOK),
    "block_output": ("layers[%s]", CONST_OUTPUT_HOOK),
    "mlp_activation": ("layers[%s].mlp.act_fn", CONST_OUTPUT_HOOK),
    "mlp_output": ("layers[%s].mlp", CONST_OUTPUT_HOOK),
    "mlp_input": ("layers[%s].mlp", CONST_INPUT_HOOK),
    "attention_value_output": ("layers[%s].self_attn.dense", CONST_INPUT_HOOK),
    "head_attention_value_output": ("layers[%s].self_attn.dense", CONST_INPUT_HOOK),
    "attention_output": ("layers[%s].self_attn", CONST_OUTPUT_HOOK),
    "attention_input": ("layers[%s].self_attn", CONST_INPUT_HOOK),
#     "query_output": ("layers[%s].self_attn.query_key_value", CONST_OUTPUT_HOOK),
#     "key_output": ("layers[%s].self_attn.query_key_value", CONST_OUTPUT_HOOK),
#     "value_output": ("layers[%s].self_attn.query_key_value", CONST_OUTPUT_HOOK),
#     "head_query_output": ("layers[%s].self_attn.query_key_value", CONST_OUTPUT_HOOK),
#     "head_key_output": ("layers[%s].self_attn.query_key_value", CONST_OUTPUT_HOOK),
#     "head_value_output": ("layers[%s].self_attn.query_key_value", CONST_OUTPUT_HOOK),
}


phi_lm_type_to_dimension_mapping = {
    "block_input": ("hidden_size",),
    "block_output": ("hidden_size",),
    "mlp_activation": ("intermediate_size",),
    "mlp_output": ("hidden_size",),
    "mlp_input": ("hidden_size",),
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
}