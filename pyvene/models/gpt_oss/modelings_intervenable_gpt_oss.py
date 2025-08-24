"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.

We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.
"""

from ..constants import *


"""gpt-oss base model"""
gpt_oss_type_to_module_mapping = {
    "block_input": ("layers[%s]", CONST_INPUT_HOOK),
    "block_output": ("layers[%s]", CONST_OUTPUT_HOOK),
    "mlp_input": ("layers[%s].mlp", CONST_INPUT_HOOK),
    "mlp_output": ("layers[%s].mlp", CONST_OUTPUT_HOOK),
    "router_input": ("layers[%s].mlp.router", CONST_INPUT_HOOK),
    "router_output": ("layers[%s].mlp.router", CONST_OUTPUT_HOOK),
    "expert_input": ("layers[%s].mlp.experts", CONST_INPUT_HOOK),
    "expert_output": ("layers[%s].mlp.experts", CONST_OUTPUT_HOOK),
    "attention_input": ("layers[%s].self_attn", CONST_INPUT_HOOK),
    "attention_output": ("layers[%s].self_attn", CONST_OUTPUT_HOOK),
    "attention_value_output": ("layers[%s].self_attn.o_proj", CONST_INPUT_HOOK),
    "head_attention_value_output": (
        "layers[%s].self_attn.o_proj",
        CONST_INPUT_HOOK,
        (split_head_and_permute, "num_attention_heads"),
    ),
    "query_output": ("layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK),
    "key_output": ("layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK),
    "value_output": ("layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK),
    "head_query_output": (
        "layers[%s].self_attn.q_proj",
        CONST_OUTPUT_HOOK,
        (split_head_and_permute, "num_attention_heads"),
    ),
    "head_key_output": (
        "layers[%s].self_attn.k_proj",
        CONST_OUTPUT_HOOK,
        (split_head_and_permute, "num_key_value_heads"),
    ),
    "head_value_output": (
        "layers[%s].self_attn.v_proj",
        CONST_OUTPUT_HOOK,
        (split_head_and_permute, "num_key_value_heads"),
    ),
}


gpt_oss_type_to_dimension_mapping = {
    "num_attention_heads": ("num_attention_heads",),
    "num_key_value_heads": ("num_key_value_heads",),
    "num_local_experts": ("num_local_experts",),
    "num_experts_per_tok": ("num_experts_per_tok",),
    "block_input": ("hidden_size",),
    "block_output": ("hidden_size",),
    "mlp_input": ("hidden_size",),
    "mlp_output": ("hidden_size",),
    "router_input": ("hidden_size",),
    "router_output": ("num_local_experts",),
    "expert_input": ("hidden_size",),
    "expert_output": ("hidden_size",),
    "attention_input": ("hidden_size",),
    "attention_output": ("hidden_size",),
    "attention_value_output": ("hidden_size",),
    "head_attention_value_output": ("hidden_size/num_attention_heads",),
    "query_output": ("hidden_size",),
    "key_output": ("hidden_size",),
    "value_output": ("hidden_size",),
    "head_query_output": ("hidden_size/num_attention_heads",),
    "head_key_output": ("hidden_size/num_key_value_heads",),
    "head_value_output": ("hidden_size/num_key_value_heads",),
}


"""gpt-oss model with LM head"""
gpt_oss_lm_type_to_module_mapping = {}
for k, v in gpt_oss_type_to_module_mapping.items():
    gpt_oss_lm_type_to_module_mapping[k] = (f"model.{v[0]}",) + v[1:]

gpt_oss_lm_type_to_dimension_mapping = gpt_oss_type_to_dimension_mapping


def create_gpt_oss(name="openai/gpt-oss-20b", cache_dir=None, access_token=None):
    """Creates a GPT-OSS model, config, and tokenizer from the given name and revision"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    config = AutoConfig.from_pretrained(name, cache_dir=cache_dir, token=access_token)
    tokenizer = AutoTokenizer.from_pretrained(
        name, cache_dir=cache_dir, token=access_token
    )
    gpt_oss = AutoModelForCausalLM.from_pretrained(
        name, cache_dir=cache_dir, token=access_token
    )
    print("loaded model")
    return config, tokenizer, gpt_oss
