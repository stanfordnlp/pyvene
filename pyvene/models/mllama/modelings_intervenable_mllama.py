"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.

We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.

Author: Alexa Tartaglini
"""


import torch
from ..constants import *


mllama_type_to_module_mapping = {
    "language.block_input": ("language_model.model.layers[%s]", CONST_INPUT_HOOK),
    "language.block_output": ("language_model.model.layers[%s]", CONST_OUTPUT_HOOK),
    "language.mlp_activation": ("language_model.model.layers[%s].mlp.act_fn", CONST_OUTPUT_HOOK),
    "language.mlp_output": ("language_model.model.layers[%s].mlp", CONST_OUTPUT_HOOK),
    "language.mlp_input": ("language_model.model.layers[%s].mlp", CONST_INPUT_HOOK),
    "language.attention_value_output": ("language_model.model.layers[%s].self_attn.o_proj", CONST_INPUT_HOOK),
    "language.head_attention_value_output": ("language_model.model.layers[%s].self_attn.o_proj", CONST_INPUT_HOOK, (split_head_and_permute, "language.n_head")),
    "language.attention_output": ("language_model.model.layers[%s].self_attn", CONST_OUTPUT_HOOK),
    "language.attention_input": ("language_model.model.layers[%s].self_attn", CONST_INPUT_HOOK),
    "language.query_output": ("language_model.model.layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK),
    "language.key_output": ("language_model.model.layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK),
    "language.value_output": ("language_model.model.layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK),
    "language.head_query_output": ("language_model.model.layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "language.n_head")),
    "language.head_key_output": ("language_model.model.layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "language.n_kv_head")),
    "language.head_value_output": ("language_model.model.layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "language.n_kv_head")),
    "vision.block_input": ("vision_model.transformer.layers[%s]", CONST_INPUT_HOOK),
    "vision.block_output": ("vision_model.transformer.layers[%s]", CONST_OUTPUT_HOOK),
    "vision.mlp_activation": ("vision_model.transformer.layers[%s].mlp.act_fn", CONST_OUTPUT_HOOK),
    "vision.mlp_output": ("vision_model.transformer.layers[%s].mlp", CONST_OUTPUT_HOOK),
    "vision.mlp_input": ("vision_model.transformer.layers[%s].mlp", CONST_INPUT_HOOK),
    "vision.attention_value_output": ("vision_model.transformer.layers[%s].self_attn.o_proj", CONST_INPUT_HOOK),
    "vision.head_attention_value_output": ("vision_model.transformer.layers[%s].self_attn.o_proj", CONST_INPUT_HOOK, (split_head_and_permute, "vision.n_head")),
    "vision.attention_output": ("vision_model.transformer.layers[%s].self_attn", CONST_OUTPUT_HOOK),
    "vision.attention_input": ("vision_model.transformer.layers[%s].self_attn", CONST_INPUT_HOOK),
    "vision.query_output": ("vision_model.transformer.layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK),
    "vision.key_output": ("vision_model.transformer.layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK),
    "vision.value_output": ("vision_model.transformer.layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK),
    "vision.head_query_output": ("vision_model.transformer.layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "vision.n_head")),
    "vision.head_key_output": ("vision_model.transformer.layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "vision.n_head")),
    "vision.head_value_output": ("vision_model.transformer.layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "vision.n_head")),
    "global.block_input": ("vision_model.global_transformer.layers[%s]", CONST_INPUT_HOOK),
    "global.block_output": ("vision_model.global_transformer.layers[%s]", CONST_OUTPUT_HOOK),
    "global.mlp_activation": ("vision_model.global_transformer.layers[%s].mlp.act_fn", CONST_OUTPUT_HOOK),
    "global.mlp_output": ("vision_model.global_transformer.layers[%s].mlp", CONST_OUTPUT_HOOK),
    "global.mlp_input": ("vision_model.global_transformer.layers[%s].mlp", CONST_INPUT_HOOK),
    "global.attention_value_output": ("vision_model.global_transformer.layers[%s].self_attn.o_proj", CONST_INPUT_HOOK),
    "global.head_attention_value_output": ("vision_model.global_transformer.layers[%s].self_attn.o_proj", CONST_INPUT_HOOK, (split_head_and_permute, "global.n_head")),
    "global.attention_output": ("vision_model.global_transformer.layers[%s].self_attn", CONST_OUTPUT_HOOK),
    "global.attention_input": ("vision_model.global_transformer.layers[%s].self_attn", CONST_INPUT_HOOK),
    "global.query_output": ("vision_model.global_transformer.layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK),
    "global.key_output": ("vision_model.global_transformer.layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK),
    "global.value_output": ("vision_model.global_transformer.layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK),
    "global.head_query_output": ("vision_model.global_transformer.layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "global.n_head")),
    "global.head_key_output": ("vision_model.global_transformer.layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "global.n_head")),
    "global.head_value_output": ("vision_model.global_transformer.layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "global.n_head")),
}


mllama_type_to_dimension_mapping = {
    "language.n_head": ("text_config.num_attention_heads",),
    "language.n_kv_head": ("text_config.num_key_value_heads",),
    "language.block_input": ("text_config.hidden_size",),
    "language.block_output": ("text_config.hidden_size",),
    "language.mlp_activation": ("text_config.intermediate_size",),
    "language.mlp_output": ("text_config.hidden_size",),
    "language.mlp_input": ("text_config.hidden_size",),
    "language.attention_value_output": ("text_config.hidden_size",),
    "language.head_attention_value_output": ("text_config.hidden_size/text_config.num_attention_heads",),
    "language.attention_output": ("text_config.hidden_size",),
    "language.attention_input": ("text_config.hidden_size",),
    "language.query_output": ("text_config.hidden_size",),
    "language.key_output": ("text_config.hidden_size",),
    "language.value_output": ("text_config.hidden_size",),
    "language.head_query_output": ("text_config.hidden_size/text_config.num_attention_heads",),
    "language.head_key_output": ("text_config.hidden_size/text_config.num_attention_heads",),
    "language.head_value_output": ("text_config.hidden_size/text_config.num_attention_heads",),
    "vision.n_head": ("vision_config.num_attention_heads",),
    "vision.n_kv_head": ("vision_config.num_key_value_heads",),
    "vision.block_input": ("vision_config.hidden_size",),
    "vision.block_output": ("vision_config.hidden_size",),
    "vision.mlp_activation": ("vision_config.intermediate_size",),
    "vision.mlp_output": ("vision_config.hidden_size",),
    "vision.mlp_input": ("vision_config.hidden_size",),
    "vision.attention_value_output": ("vision_config.hidden_size",),
    "vision.head_attention_value_output": ("vision_config.hidden_size/vision_config.num_attention_heads",),
    "vision.attention_output": ("vision_config.hidden_size",),
    "vision.attention_input": ("vision_config.hidden_size",),
    "vision.query_output": ("vision_config.hidden_size",),
    "vision.key_output": ("vision_config.hidden_size",),
    "vision.value_output": ("vision_config.hidden_size",),
    "vision.head_query_output": ("vision_config.hidden_size/vision_config.num_attention_heads",),
    "vision.head_key_output": ("vision_config.hidden_size/vision_config.num_attention_heads",),
    "vision.head_value_output": ("vision_config.hidden_size/vision_config.num_attention_heads",),
    "global.n_head": ("vision_config.num_attention_heads",),
    "global.n_kv_head": ("vision_config.num_key_value_heads",),
    "global.block_input": ("vision_config.hidden_size",),
    "global.block_output": ("vision_config.hidden_size",),
    "global.mlp_activation": ("vision_config.intermediate_size",),
    "global.mlp_output": ("vision_config.hidden_size",),
    "global.mlp_input": ("vision_config.hidden_size",),
    "global.attention_value_output": ("vision_config.hidden_size",),
    "global.head_attention_value_output": ("vision_config.hidden_size/vision_config.num_attention_heads",),
    "global.attention_output": ("vision_config.hidden_size",),
    "global.attention_input": ("vision_config.hidden_size",),
    "global.query_output": ("vision_config.hidden_size",),
    "global.key_output": ("vision_config.hidden_size",),
    "global.value_output": ("vision_config.hidden_size",),
    "global.head_query_output": ("vision_config.hidden_size/vision_config.num_attention_heads",),
    "global.head_key_output": ("vision_config.hidden_size/vision_config.num_attention_heads",),
    "global.head_value_output": ("vision_config.hidden_size/vision_config.num_attention_heads",),
}

"""mllama model with LM head"""
mllama_lm_type_to_module_mapping = {}
for k, v in mllama_type_to_module_mapping.items():
    mllama_lm_type_to_module_mapping[k] = (f"model.{v[0]}", ) + v[1:]
mllama_lm_type_to_dimension_mapping = mllama_type_to_dimension_mapping

"""llama model with classifier head"""
mllama_classifier_type_to_module_mapping = {}
for k, v in mllama_type_to_module_mapping.items():
    mllama_classifier_type_to_module_mapping[k] = (f"model.{v[0]}", ) + v[1:]
mllama_classifier_type_to_dimension_mapping = mllama_type_to_dimension_mapping

def create_mllama(
    name="meta-llama/Llama-3.2-11B-Vision-Instruct", cache_dir=None, dtype=torch.bfloat16, config=None
):
    """Creates a LLaMA Causal LM model, config, and tokenizer from the given name and revision"""
    from transformers import MllamaForConditionalGeneration, AutoProcessor, MllamaConfig
    if config is None:
        config = MllamaConfig.from_pretrained(name, cache_dir=cache_dir)
        mllama = MllamaForConditionalGeneration.from_pretrained(
            name,
            config=config,
            cache_dir=cache_dir,
            torch_dtype=dtype,  # save memory
        )
        tokenizer = AutoProcessor.from_pretrained(name, cache_dir=cache_dir)
    else:
        mllama = MllamaForConditionalGeneration(config)
        tokenizer = AutoProcessor.from_pretrained(name, cache_dir=cache_dir)
    print("loaded model")
    return config, tokenizer, mllama