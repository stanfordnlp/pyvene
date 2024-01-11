CONST_VALID_INTERVENABLE_UNIT = ["pos", "h", "h.pos", "t"]


CONST_INPUT_HOOK = "register_forward_pre_hook"
CONST_OUTPUT_HOOK = "register_forward_hook"


CONST_TRANSFORMER_TOPOLOGICAL_ORDER = [
    "block_input",
    "query_output",
    "head_query_output",
    "key_output",
    "head_key_output",
    "value_output",
    "head_value_output",
    "attention_input",
    "head_attention_value_output",
    "attention_value_output",
    "attention_output",
    "cross_attention_input",
    "head_cross_attention_value_output",
    "cross_attention_value_output",
    "cross_attention_output",
    "mlp_input",
    "mlp_activation",
    "mlp_output",
    "block_output",
]


CONST_MLP_TOPOLOGICAL_ORDER = [
    "block_input",
    "mlp_activation",
    "block_output",
]


CONST_GRU_TOPOLOGICAL_ORDER = [
    "cell_input",
    "x2h_output",
    "h2h_output",
    "reset_x2h_output",
    "update_x2h_output",
    "new_x2h_output",
    "reset_h2h_output",
    "update_h2h_output",
    "new_h2h_output",
    "reset_gate_input",
    "update_gate_input",
    "new_gate_input",
    "reset_gate_output",
    "update_gate_output",
    "new_gate_output",
    "cell_output",
]


CONST_QKV_INDICES = {
    "query_output": 0,
    "key_output": 1,
    "value_output": 2,
    "head_query_output": 0,
    "head_key_output": 1,
    "head_value_output": 2,
    "reset_x2h_output": 0,
    "update_x2h_output": 1,
    "new_x2h_output": 2,
    "reset_h2h_output": 0,
    "update_h2h_output": 1,
    "new_h2h_output": 2,
}

CONST_RUN_INDICES = {
    "reset_x2h_output": 0,
    "update_x2h_output": 1,
    "new_x2h_output": 2,
    "reset_h2h_output": 0,
    "update_h2h_output": 1,
    "new_h2h_output": 2,
}
