"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.

We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.
"""


from ..constants import CONST_INPUT_HOOK, CONST_OUTPUT_HOOK

xlm_generic_type_to_module_mapping = {
    "block_input": ("transformer.attentions[%s]", CONST_INPUT_HOOK),
    "block_output": ("transformer.layer_norm2[%s]", CONST_OUTPUT_HOOK),
    "mlp_activation": ("transformer.ffns[%s].lin2", CONST_INPUT_HOOK),
    "mlp_output": ("transformer.ffns[%s].lin2", CONST_OUTPUT_HOOK),
    "mlp_input": ("transformer.ffns[%s].lin1", CONST_INPUT_HOOK),
    "attention_value_output": ("transformer.attentions[%s].out_lin", CONST_INPUT_HOOK),
    "head_attention_value_output": ("transformer.attentions[%s].out_lin", CONST_INPUT_HOOK),
    "attention_output": ("transformer.attentions[%s].out_lin", CONST_OUTPUT_HOOK),
    "attention_input": ("transformer.attentions[%s]", CONST_INPUT_HOOK),
    "query_output": ("transformer.attentions[%s].q_lin", CONST_OUTPUT_HOOK),
    "key_output": ("transformer.attentions[%s].k_lin", CONST_OUTPUT_HOOK),
    "value_output": ("transformer.attentions[%s].v_lin", CONST_OUTPUT_HOOK),
    "head_query_output": ("transformer.attentions[%s].q_lin", CONST_OUTPUT_HOOK),
    "head_key_output": ("transformer.attentions[%s].k_lin", CONST_OUTPUT_HOOK),
    "head_value_output": ("transformer.attentions[%s].v_lin", CONST_OUTPUT_HOOK),
}

xlm_generic_type_to_dimension_mapping = {
    "block_input": ("emb_dim",),
    "block_output": ("emb_dim",),
    "mlp_activation": (
        "emb_dim*4",
    ),
    "mlp_output": ("emb_dim",),
    "mlp_input": ("emb_dim",),
    "attention_value_output": ("emb_dim",),
    "head_attention_value_output": ("emb_dim/n_heads",),
    "attention_output": ("emb_dim",),
    "attention_input": ("emb_dim",),
    "query_output": ("emb_dim",),
    "key_output": ("emb_dim",),
    "value_output": ("emb_dim",),
    "head_query_output": ("emb_dim/n_heads",),
    "head_key_output": ("emb_dim/n_heads",),
    "head_value_output": ("emb_dim/n_heads",),
}