"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.

We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.
"""


from models.constants import CONST_INPUT_HOOK, CONST_OUTPUT_HOOK


"""gpt2 base model"""
gpt2_type_to_module_mapping = {
    'block_input': ("h[%s]", CONST_INPUT_HOOK), 
    'block_output': ("h[%s]", CONST_OUTPUT_HOOK), 
    'mlp_activation': ("h[%s].mlp.act", CONST_OUTPUT_HOOK), 
    'mlp_output': ("h[%s].mlp", CONST_OUTPUT_HOOK), 
    'mlp_input': ("h[%s].mlp", CONST_INPUT_HOOK), 
    'attention_value_output': ("h[%s].attn.c_proj", CONST_INPUT_HOOK),
    'head_attention_value_output': ("h[%s].attn.c_proj", CONST_INPUT_HOOK),
    'attention_output': ("h[%s].attn", CONST_OUTPUT_HOOK),
    'attention_input': ("h[%s].attn", CONST_INPUT_HOOK),
    'query_output': ("h[%s].attn.c_attn", CONST_OUTPUT_HOOK),
    'key_output': ("h[%s].attn.c_attn", CONST_OUTPUT_HOOK),
    'value_output': ("h[%s].attn.c_attn", CONST_OUTPUT_HOOK),
    'head_query_output': ("h[%s].attn.c_attn", CONST_OUTPUT_HOOK),
    'head_key_output': ("h[%s].attn.c_attn", CONST_OUTPUT_HOOK),
    'head_value_output': ("h[%s].attn.c_attn", CONST_OUTPUT_HOOK),
}


gpt2_type_to_dimension_mapping = {
    'block_input': ("config.n_embd", ), 
    'block_output': ("config.n_embd", ), 
    'mlp_activation': ("config.n_inner", "config.n_embd*4", ), 
    'mlp_output': ("config.n_embd", ), 
    'mlp_input': ("config.n_embd", ), 
    'attention_value_output': ("config.n_embd", ),
    'head_attention_value_output': ("config.n_embd/config.n_head", ),
    'attention_output': ("config.n_embd", ),
    'attention_input': ("config.n_embd", ),
    'query_output': ("config.n_embd", ),
    'key_output': ("config.n_embd", ),
    'value_output': ("config.n_embd", ),
    'head_query_output': ("config.n_embd/config.n_head", ),
    'head_key_output': ("config.n_embd/config.n_head", ),
    'head_value_output': ("config.n_embd/config.n_head", ),
}


"""gpt2 model with LM head"""
gpt2_lm_type_to_module_mapping = {}
for k, v in gpt2_type_to_module_mapping.items():
    gpt2_lm_type_to_module_mapping[k] = (f"transformer.{v[0]}", v[1])


gpt2_lm_type_to_dimension_mapping = gpt2_type_to_dimension_mapping

