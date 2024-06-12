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

llava_type_to_module_mapping = {
    "block_input": ("language_model.model.layers[%s]", CONST_INPUT_HOOK),
    "block_output": ("language_model.model.layers[%s]", CONST_OUTPUT_HOOK),
    "mlp_activation": ("language_model.model.layers[%s].mlp.act_fn", CONST_OUTPUT_HOOK),
    "mlp_output": ("language_model.model.layers[%s].mlp", CONST_OUTPUT_HOOK),
    "mlp_input": ("language_model.model.layers[%s].mlp", CONST_INPUT_HOOK),
    "attention_value_output": ("language_model.model.layers[%s].self_attn.o_proj", CONST_INPUT_HOOK),
    "head_attention_value_output": ("language_model.model.layers[%s].self_attn.o_proj", CONST_INPUT_HOOK, (split_head_and_permute, "n_head")),
    "attention_output": ("language_model.model.layers[%s].self_attn", CONST_OUTPUT_HOOK),
    "attention_input": ("language_model.model.layers[%s].self_attn", CONST_INPUT_HOOK),
    "query_output": ("language_model.model.layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK),
    "key_output": ("language_model.model.layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK),
    "value_output": ("language_model.model.layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK),
    "head_query_output": ("language_model.model.layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "n_head")),
    "head_key_output": ("language_model.model.layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "n_kv_head")),
    "head_value_output": ("language_model.model.layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "n_kv_head")),
}


llava_type_to_dimension_mapping = {
    "n_head": ("text_config.num_attention_heads",),
    "n_kv_head": ("text_config.num_key_value_heads",),
    "block_input": ("text_config.hidden_size",),
    "block_output": ("text_config.hidden_size",),
    "mlp_activation": ("text_config.intermediate_size",),
    "mlp_output": ("text_config.hidden_size",),
    "mlp_input": ("text_config.hidden_size",),
    "attention_value_output": ("text_config.hidden_size",),
    "head_attention_value_output": ("text_config.hidden_size/text_config.num_attention_heads",),
    "attention_output": ("text_config.hidden_size",),
    "attention_input": ("text_config.hidden_size",),
    "query_output": ("text_config.hidden_size",),
    "key_output": ("text_config.hidden_size",),
    "value_output": ("text_config.hidden_size",),
    "head_query_output": ("text_config.hidden_size/text_config.num_attention_heads",),
    "head_key_output": ("text_config.hidden_size/text_config.num_attention_heads",),
    "head_value_output": ("text_config.hidden_size/text_config.num_attention_heads",),
}


"""llava model with LM head"""
llava_lm_type_to_module_mapping = {}
for k, v in llava_type_to_module_mapping.items():
    llava_lm_type_to_module_mapping[k] = (f"model.{v[0]}", ) + v[1:]


llava_lm_type_to_dimension_mapping = llava_type_to_dimension_mapping


"""llava model with classifier head"""
llava_classifier_type_to_module_mapping = {}
for k, v in llava_type_to_module_mapping.items():
    llava_classifier_type_to_module_mapping[k] = (f"model.{v[0]}", ) + v[1:]


llava_classifier_type_to_dimension_mapping = llava_type_to_dimension_mapping




def create_llava(
    name="llava-hf/llava-1.5-7b-hf", cache_dir=None, dtype=torch.bfloat16
):
    """Creates a llava Causal LM model, config, and tokenizer from the given name and revision"""
    from transformers import LlavaForConditionalGeneration, LlavaConfig, AutoTokenizer, AutoProcessor

    config = LlavaConfig.from_pretrained(name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False)
    llava = LlavaForConditionalGeneration.from_pretrained(
        name,
        config=config,
        cache_dir=cache_dir,
        torch_dtype=dtype,
    )

    image_processor = AutoProcessor.from_pretrained(name)

    print("loaded model")
    return config, tokenizer, llava, image_processor

