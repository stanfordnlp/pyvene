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


llama_type_to_module_mapping = {
    "block_input": ("layers[%s]", CONST_INPUT_HOOK),
    "block_output": ("layers[%s]", CONST_OUTPUT_HOOK),
    "mlp_activation": ("layers[%s].mlp.act_fn", CONST_OUTPUT_HOOK),
    "mlp_output": ("layers[%s].mlp", CONST_OUTPUT_HOOK),
    "mlp_input": ("layers[%s].mlp", CONST_INPUT_HOOK),
    "attention_value_output": ("layers[%s].self_attn.o_proj", CONST_INPUT_HOOK),
    "head_attention_value_output": ("layers[%s].self_attn.o_proj", CONST_INPUT_HOOK, (split_head_and_permute, "n_head")),
    "attention_output": ("layers[%s].self_attn", CONST_OUTPUT_HOOK),
    "attention_input": ("layers[%s].self_attn", CONST_INPUT_HOOK),
    "query_output": ("layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK),
    "key_output": ("layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK),
    "value_output": ("layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK),
    "head_query_output": ("layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "n_head")),
    "head_key_output": ("layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "n_kv_head")),
    "head_value_output": ("layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "n_kv_head")),
}


llama_type_to_dimension_mapping = {
    "n_head": ("num_attention_heads",),
    "n_kv_head": ("num_key_value_heads",),
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
llama_lm_type_to_module_mapping = {}
for k, v in llama_type_to_module_mapping.items():
    llama_lm_type_to_module_mapping[k] = (f"model.{v[0]}", ) + v[1:]


llama_lm_type_to_dimension_mapping = llama_type_to_dimension_mapping


"""llama model with classifier head"""
llama_classifier_type_to_module_mapping = {}
for k, v in llama_type_to_module_mapping.items():
    llama_classifier_type_to_module_mapping[k] = (f"model.{v[0]}", ) + v[1:]


llama_classifier_type_to_dimension_mapping = llama_type_to_dimension_mapping


def create_llama(
    name="sharpbai/alpaca-7b-merged", cache_dir=None, dtype=torch.bfloat16, config=None
):
    """Creates a LLaMA Causal LM model, config, and tokenizer from the given name and revision"""
    from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
    if config is None:
        config = LlamaConfig.from_pretrained(name, cache_dir=cache_dir)
        llama = LlamaForCausalLM.from_pretrained(
            name,
            config=config,
            cache_dir=cache_dir,
            torch_dtype=dtype,  # save memory
        )
        tokenizer = LlamaTokenizer.from_pretrained(name, cache_dir=cache_dir)
    else:
        llama = LlamaForCausalLM(config)
        tokenizer = LlamaTokenizer.from_pretrained(name, cache_dir=cache_dir)
    print("loaded model")
    return config, tokenizer, llama