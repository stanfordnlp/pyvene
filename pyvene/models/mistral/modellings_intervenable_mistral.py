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


mistral_type_to_module_mapping = {
    "block_input": ("layers[%s]", CONST_INPUT_HOOK),
    "block_output": ("layers[%s]", CONST_OUTPUT_HOOK),
    "mlp_activation": ("layers[%s].mlp.act_fn", CONST_OUTPUT_HOOK),
    "mlp_output": ("layers[%s].mlp", CONST_OUTPUT_HOOK),
    "mlp_input": ("layers[%s].mlp", CONST_INPUT_HOOK),
    "attention_value_output": ("layers[%s].self_attn.o_proj", CONST_INPUT_HOOK),
    "head_attention_value_output": ("layers[%s].self_attn.o_proj", CONST_INPUT_HOOK),
    "attention_output": ("layers[%s].self_attn", CONST_OUTPUT_HOOK),
    "attention_input": ("layers[%s].self_attn", CONST_INPUT_HOOK),
    "query_output": ("layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK),
    "key_output": ("layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK),
    "value_output": ("layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK),
    "head_query_output": ("layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK),
    "head_key_output": ("layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK),
    "head_value_output": ("layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK),
}


mistral_type_to_dimension_mapping = {
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


"""llama model with LM head"""
mistral_lm_type_to_module_mapping = {}
for k, v in mistral_type_to_module_mapping.items():
    mistral_lm_type_to_module_mapping[k] = (f"model.{v[0]}", v[1])


mistral_lm_type_to_dimension_mapping = mistral_type_to_dimension_mapping


def create_mistral(
    name="mistralai/Mistral-7B-v0.1", cache_dir=None
):
    """Creates a Mistral Causal LM model, config, and tokenizer from the given name and revision"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    config = AutoConfig.from_pretrained(name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    llama = AutoModelForCausalLM.from_pretrained(
        name,
        config=config,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,  # save memory
    )
    print("loaded model")
    return config, tokenizer, llama
