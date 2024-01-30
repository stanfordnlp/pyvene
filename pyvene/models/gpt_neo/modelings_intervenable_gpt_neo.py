"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.

We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.
"""


from ..constants import *


"""gpt_neo base model"""
gpt_neo_type_to_module_mapping = {
    "block_input": ("h[%s]", CONST_INPUT_HOOK),
    "block_output": ("h[%s]", CONST_OUTPUT_HOOK),
    "mlp_activation": ("h[%s].mlp.act", CONST_OUTPUT_HOOK),
    "mlp_output": ("h[%s].mlp", CONST_OUTPUT_HOOK),
    "mlp_input": ("h[%s].mlp", CONST_INPUT_HOOK),
    "attention_value_output": ("h[%s].attn.out_proj", CONST_INPUT_HOOK),
    "head_attention_value_output": ("h[%s].attn.out_proj", CONST_INPUT_HOOK, (split_head_and_permute, "n_head")),
    "attention_output": ("h[%s].attn", CONST_OUTPUT_HOOK),
    "attention_input": ("h[%s].attn", CONST_INPUT_HOOK),
    "query_output": ("h[%s].attn.q_proj", CONST_OUTPUT_HOOK),
    "key_output": ("h[%s].attn.k_proj", CONST_OUTPUT_HOOK),
    "value_output": ("h[%s].attn.v_proj", CONST_OUTPUT_HOOK),
    "head_query_output": ("h[%s].attn.q_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "n_head")),
    "head_key_output": ("h[%s].attn.k_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "n_head")),
    "head_value_output": ("h[%s].attn.v_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "n_head")),
}


gpt_neo_type_to_dimension_mapping = {
    "n_head": "num_heads",
    "block_input": ("hidden_size",),
    "block_output": ("hidden_size",),
    "mlp_activation": (
        "intermediate_size",
        "hidden_size*4",
    ),
    "mlp_output": ("hidden_size",),
    "mlp_input": ("hidden_size",),
    "attention_value_output": ("hidden_size",),
    "head_attention_value_output": ("hidden_size/num_heads",),
    "attention_output": ("hidden_size",),
    "attention_input": ("hidden_size",),
    "query_output": ("hidden_size",),
    "key_output": ("hidden_size",),
    "value_output": ("hidden_size",),
    "head_query_output": ("hidden_size/num_heads",),
    "head_key_output": ("hidden_size/num_heads",),
    "head_value_output": ("hidden_size/num_heads",),
}


"""gpt_neo model with LM head"""
gpt_neo_lm_type_to_module_mapping = {}
for k, v in gpt_neo_type_to_module_mapping.items():
    gpt_neo_lm_type_to_module_mapping[k] = (f"transformer.{v[0]}", v[1])


gpt_neo_lm_type_to_dimension_mapping = gpt_neo_type_to_dimension_mapping


def create_gpt_neo(
    name="roneneldan/TinyStories-33M", cache_dir=None
):
    """Creates a GPT2 model, config, and tokenizer from the given name and revision"""
    from transformers import GPTNeoForCausalLM, GPT2Tokenizer, GPTNeoConfig

    config = GPTNeoConfig.from_pretrained(name)
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")  # not sure
    gpt_neo = GPTNeoForCausalLM.from_pretrained(name)
    print("loaded model")
    return config, tokenizer, gpt_neo
