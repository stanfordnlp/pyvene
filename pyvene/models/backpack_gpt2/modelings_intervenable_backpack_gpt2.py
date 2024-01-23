"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.

We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.
"""


from ..constants import CONST_INPUT_HOOK, CONST_OUTPUT_HOOK, CONST_QKV_INDICES


"""gpt2 base model"""
backpack_gpt2_lm_type_to_module_mapping = {
    "block_input": ("backpack.gpt2_model.h[%s]", CONST_INPUT_HOOK),
    "block_output": ("backpack.gpt2_model.h[%s]", CONST_OUTPUT_HOOK),
    "mlp_activation": ("backpack.gpt2_model.h[%s].mlp.act", CONST_OUTPUT_HOOK),
    "mlp_output": ("backpack.gpt2_model.h[%s].mlp", CONST_OUTPUT_HOOK),
    "mlp_input": ("backpack.gpt2_model.h[%s].mlp", CONST_INPUT_HOOK),
    "attention_value_output": ("backpack.gpt2_model.h[%s].attn.c_proj", CONST_INPUT_HOOK),
    "head_attention_value_output": ("backpack.gpt2_model.h[%s].attn.c_proj", CONST_INPUT_HOOK),
    "attention_output": ("backpack.gpt2_model.h[%s].attn", CONST_OUTPUT_HOOK),
    "attention_input": ("backpack.gpt2_model.h[%s].attn", CONST_INPUT_HOOK),
    "query_output": ("backpack.gpt2_model.h[%s].attn.c_attn", CONST_OUTPUT_HOOK),
    "key_output": ("backpack.gpt2_model.h[%s].attn.c_attn", CONST_OUTPUT_HOOK),
    "value_output": ("backpack.gpt2_model.h[%s].attn.c_attn", CONST_OUTPUT_HOOK),
    "head_query_output": ("backpack.gpt2_model.h[%s].attn.c_attn", CONST_OUTPUT_HOOK),
    "head_key_output": ("backpack.gpt2_model.h[%s].attn.c_attn", CONST_OUTPUT_HOOK),
    "head_value_output": ("backpack.gpt2_model.h[%s].attn.c_attn", CONST_OUTPUT_HOOK),
    "sense_output": ("backpack.sense_network", CONST_OUTPUT_HOOK),
    "sense_block_output": ("backpack.sense_network.block", CONST_OUTPUT_HOOK),
    "sense_mlp_input": ("backpack.sense_network.final_mlp", CONST_INPUT_HOOK),
    "sense_mlp_output": ("backpack.sense_network.final_mlp", CONST_OUTPUT_HOOK),
    "sense_mlp_activation": ("backpack.sense_network.final_mlp.act", CONST_OUTPUT_HOOK),
    "sense_weight_input": ("backpack.sense_weight_net", CONST_INPUT_HOOK),
    "sense_weight_output": ("backpack.sense_weight_net", CONST_OUTPUT_HOOK),
}


backpack_gpt2_lm_type_to_dimension_mapping = {
    "block_input": ("n_embd",),
    "block_output": ("n_embd",),
    "mlp_activation": (
        "n_inner",
        "n_embd*4",
    ),
    "mlp_output": ("n_embd",),
    "mlp_input": ("n_embd",),
    "attention_value_output": ("n_embd",),
    "head_attention_value_output": ("n_embd/n_head",),
    "attention_output": ("n_embd",),
    "attention_input": ("n_embd",),
    "query_output": ("n_embd",),
    "key_output": ("n_embd",),
    "value_output": ("n_embd",),
    "head_query_output": ("n_embd/n_head",),
    "head_key_output": ("n_embd/n_head",),
    "head_value_output": ("n_embd/n_head",),
    "sense_output": ("n_embd",),
    "sense_block_output": ("n_embd",),
    "sense_mlp_input": ("n_embd",),
    "sense_mlp_output": ("n_embd",),
    "num_senses": ("num_senses",),
    "sense_mlp_activation": (
        "n_inner",
        "n_embd*4",
    ),
}

def create_backpack_gpt2(name="stanfordnlp/backpack-gpt2", cache_dir=None):
    """Creates a GPT2 model, config, and tokenizer from the given name and revision"""
    # Load model directly
    from transformers import AutoTokenizer
    from pyvene.models.backpack_gpt2.modelings_backpack_gpt2 import BackpackGPT2LMHeadModel

    tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
    model = BackpackGPT2LMHeadModel.from_pretrained(name, trust_remote_code=True)
    print("loaded model")
    return model.config, tokenizer, model

