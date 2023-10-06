CONST_INPUT_HOOK = "register_forward_pre_hook"
CONST_OUTPUT_HOOK = "register_forward_hook"


CONST_TRANSFORMER_TOPOLOGICAL_ORDER = [
    'block_input',
    'query_output',
    'head_query_output',
    'key_output',
    'head_key_output',
    'value_output',
    'head_value_output',
    'attention_input',
    'head_attention_value_output',
    'attention_value_output',
    'attention_output',
    'mlp_input',
    'mlp_activation',
    'mlp_output',
    'block_output'
]


CONST_QKV_INDICES = {
    "query_output": 0, 
    "key_output": 1, 
    "value_output": 2,
    "head_query_output": 0, 
    "head_key_output": 1, 
    "head_value_output": 2
}
