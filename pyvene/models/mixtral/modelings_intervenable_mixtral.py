"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.

We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.
"""

from ..constants import CONST_INPUT_HOOK, CONST_OUTPUT_HOOK


mixtral_lm_type_to_module_mapping = {
    'block_input': ("model.layers[%s]", CONST_INPUT_HOOK), 
    'block_output': ("model.layers[%s]", CONST_OUTPUT_HOOK), 
    # 'mlp_activation': ("model.layers[%s].block_sparse_moe.experts[%s].act_fn", CONST_OUTPUT_HOOK), 
    'mlp_output': ("model.layers[%s].block_sparse_moe.experts[%s]", CONST_OUTPUT_HOOK), 
    'mlp_input': ("model.layers[%s].block_sparse_moe.experts[%s]", CONST_INPUT_HOOK), 
    'mlp_gate': ("model.layers[%s].block_sparse_moe.gate", CONST_INPUT_HOOK), 
    'attention_value_output': ("model.layers[%s].self_attn.o_proj", CONST_INPUT_HOOK),
    'head_attention_value_output': ("model.layers[%s].self_attn.o_proj", CONST_INPUT_HOOK),
    'attention_output': ("model.layers[%s].self_attn", CONST_OUTPUT_HOOK),
    'attention_input': ("model.layers[%s].self_attn", CONST_INPUT_HOOK),
    'query_output': ("model.layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK),
    'key_output': ("model.layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK),
    'value_output': ("model.layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK),
    'head_query_output': ("model.layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK),
    'head_key_output': ("model.layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK),
    'head_value_output': ("model.layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK),
}


mixtral_lm_type_to_dimension_mapping = {
    'block_input': ("config.hidden_size", ), 
    'block_output': ("config.hidden_size", ), 
    'mlp_activation': ("config.intermediate_size", ), 
    'mlp_output': ("config.hidden_size", ), 
    'mlp_input': ("config.hidden_size", ), 
    'mlp_gate': ("config.num_local_experts", ), 
    'attention_value_output': ("config.hidden_size", ),
    'head_attention_value_output': ("config.hidden_size/config.num_attention_heads", ),
    'attention_output': ("config.hidden_size", ),
    'attention_input': ("config.hidden_size", ),
    'query_output': ("config.hidden_size", ),
    'key_output': ("config.hidden_size", ),
    'value_output': ("config.hidden_size", ),
    'head_query_output': ("config.hidden_size/config.num_attention_heads", ),
    'head_key_output': ("config.hidden_size/config.num_attention_heads", ),
    'head_value_output': ("config.hidden_size/config.num_attention_heads", ),
}