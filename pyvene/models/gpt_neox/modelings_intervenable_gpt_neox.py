"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.

We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.
"""


from ..constants import *


"""gpt_neox base model"""
gpt_neox_type_to_module_mapping = {
    "block_input": ("layers[%s]", CONST_INPUT_HOOK),
    "block_output": ("layers[%s]", CONST_OUTPUT_HOOK),
    "mlp_activation": ("layers[%s].mlp.act", CONST_OUTPUT_HOOK),
    "mlp_output": ("layers[%s].mlp", CONST_OUTPUT_HOOK),
    "mlp_input": ("layers[%s].mlp", CONST_INPUT_HOOK),
    "attention_value_output": ("layers[%s].attention.dense", CONST_INPUT_HOOK),
    "head_attention_value_output": ("layers[%s].attention.dense", CONST_INPUT_HOOK, (split_head_and_permute, "n_head")),
    "attention_output": ("layers[%s].attention", CONST_OUTPUT_HOOK),
    "attention_input": ("layers[%s].attention", CONST_INPUT_HOOK),
    # 'query_output': ("layers[%s].attention.query_key_value", CONST_OUTPUT_HOOK),
    # 'key_output': ("layers[%s].attention.query_key_value", CONST_OUTPUT_HOOK),
    # 'value_output': ("layers[%s].attention.query_key_value", CONST_OUTPUT_HOOK),
    # 'head_query_output': ("layers[%s].attention.query_key_value", CONST_OUTPUT_HOOK),
    # 'head_key_output': ("layers[%s].attention.query_key_value", CONST_OUTPUT_HOOK),
    # 'head_value_output': ("layers[%s].attention.query_key_value", CONST_OUTPUT_HOOK),
}


gpt_neox_type_to_dimension_mapping = {
    "n_head": "num_attention_heads",
    "block_input": ("hidden_size",),
    "block_output": ("hidden_size",),
    "mlp_activation": (
        "intermediate_size",
        "hidden_size*4",
    ),
    "mlp_output": ("hidden_size",),
    "mlp_input": ("hidden_size",),
    "attention_value_output": ("hidden_size",),
    "head_attention_value_output": ("hidden_size/num_attention_heads",),
    "attention_output": ("hidden_size",),
    "attention_input": ("hidden_size",),
    # 'query_output': ("hidden_size", ),
    # 'key_output': ("hidden_size", ),
    # 'value_output': ("hidden_size", ),
    # 'head_query_output': ("hidden_size/num_attention_heads", ),
    # 'head_key_output': ("hidden_size/num_attention_heads", ),
    # 'head_value_output': ("hidden_size/num_attention_heads", ),
}


"""gpt_neox model with LM head"""
gpt_neox_lm_type_to_module_mapping = {}
for k, v in gpt_neox_type_to_module_mapping.items():
    gpt_neox_lm_type_to_module_mapping[k] = (f"gpt_neox.{v[0]}", v[1])


gpt_neox_lm_type_to_dimension_mapping = gpt_neox_type_to_dimension_mapping


def create_gpt_neox(name="EleutherAI/pythia-70m", cache_dir=None):
    """Creates a GPT2 model, config, and tokenizer from the given name and revision"""
    from transformers import GPTNeoXForCausalLM, AutoTokenizer, GPTNeoXConfig

    config = GPTNeoXConfig.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    gpt_neox = GPTNeoXForCausalLM.from_pretrained(name)
    print("loaded model")
    return config, tokenizer, gpt_neox
