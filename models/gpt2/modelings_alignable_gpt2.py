"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.

We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.

There are two customized functions that need to be defined
in order to do head-level interventions:
- _output_to_subcomponent
- _scatter_intervention_output
"""


from models.constants import CONST_INPUT_HOOK, CONST_OUTPUT_HOOK, CONST_QKV_INDICES


"""gpt2 base model"""
gpt2_type_to_module_mapping = {
    'block_input': ("h[%s]", CONST_INPUT_HOOK), 
    'block_output': ("h[%s]", CONST_OUTPUT_HOOK), 
    'mlp_activation': ("h[%s].mlp.act", CONST_OUTPUT_HOOK), 
    'mlp_output': ("h[%s].mlp", CONST_OUTPUT_HOOK), 
    'mlp_input': ("h[%s].mlp", CONST_INPUT_HOOK), 
    'attention_value_output': ("h[%s].attn.c_proj", CONST_INPUT_HOOK),
    'head_attention_value_output': ("h[%s].attn.c_proj", CONST_INPUT_HOOK),
    'attention_output': ("h[%s].attn", CONST_OUTPUT_HOOK),
    'attention_input': ("h[%s].attn", CONST_INPUT_HOOK),
    'query_output': ("h[%s].attn.c_attn", CONST_OUTPUT_HOOK),
    'key_output': ("h[%s].attn.c_attn", CONST_OUTPUT_HOOK),
    'value_output': ("h[%s].attn.c_attn", CONST_OUTPUT_HOOK),
    'head_query_output': ("h[%s].attn.c_attn", CONST_OUTPUT_HOOK),
    'head_key_output': ("h[%s].attn.c_attn", CONST_OUTPUT_HOOK),
    'head_value_output': ("h[%s].attn.c_attn", CONST_OUTPUT_HOOK),
}


gpt2_type_to_dimension_mapping = {
    'block_input': ("config.n_embd", ), 
    'block_output': ("config.n_embd", ), 
    'mlp_activation': ("config.n_inner", "config.n_embd*4", ), 
    'mlp_output': ("config.n_embd", ), 
    'mlp_input': ("config.n_embd", ), 
    'attention_value_output': ("config.n_embd", ),
    'head_attention_value_output': ("config.n_embd/config.n_head", ),
    'attention_output': ("config.n_embd", ),
    'attention_input': ("config.n_embd", ),
    'query_output': ("config.n_embd", ),
    'key_output': ("config.n_embd", ),
    'value_output': ("config.n_embd", ),
    'head_query_output': ("config.n_embd/config.n_head", ),
    'head_key_output': ("config.n_embd/config.n_head", ),
    'head_value_output': ("config.n_embd/config.n_head", ),
}


"""gpt2 model with LM head"""
gpt2_lm_type_to_module_mapping = {}
for k, v in gpt2_type_to_module_mapping.items():
    gpt2_lm_type_to_module_mapping[k] = (f"transformer.{v[0]}", v[1])


gpt2_lm_type_to_dimension_mapping = gpt2_type_to_dimension_mapping


def create_gpt2(name="gpt2", cache_dir="../.huggingface_cache"):
    """Creates a GPT2 model, config, and tokenizer from the given name and revision"""
    from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
    
    config = GPT2Config.from_pretrained(name)
    tokenizer = GPT2Tokenizer.from_pretrained(name)
    gpt = GPT2Model.from_pretrained(name, config=config, cache_dir=cache_dir)
    print("loaded model")
    return config, tokenizer, gpt


def split_heads(tensor, num_heads, attn_head_size):
    """
    Splits hidden_size dim into attn_head_size and num_heads
    """
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(new_shape)
    return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)


def gpt2_output_to_subcomponent(
    output, alignable_representation_type, model_config
):
    n_embd = model_config.n_embd
    num_heads = model_config.n_head
    attn_head_size = n_embd // num_heads

    if alignable_representation_type in {
        "query_output", "key_output", "value_output",
        "head_query_output", "head_key_output", "head_value_output",
    }:
        qkv = output.split(
            n_embd, 
            dim=2
        )
        if alignable_representation_type in {
            "head_query_output", "head_key_output", "head_value_output",
        }:
            qkv = (
                split_heads(qkv[0], num_heads, attn_head_size),
                split_heads(qkv[1], num_heads, attn_head_size),
                split_heads(qkv[2], num_heads, attn_head_size),
            ) # each with (batch, head, seq_length, head_features)
        return qkv[CONST_QKV_INDICES[alignable_representation_type]]
    elif alignable_representation_type in {"head_attention_value_output"}:
        return split_heads(output, num_heads, attn_head_size)
    else:
        return output


def gpt2_scatter_intervention_output(
    original_output, intervened_representation,
    alignable_representation_type,
    unit_locations, model_config
):
    n_embd = model_config.n_embd
    num_heads = model_config.n_head
    attn_head_size = n_embd // num_heads
    if alignable_representation_type in {
        "query_output", "key_output", "value_output",

    }:
        # replacing [b, s, 3*d] with [b, num_int, d]
        start_index = CONST_QKV_INDICES[alignable_representation_type]*n_embd
        end_index = (CONST_QKV_INDICES[alignable_representation_type]+1)*n_embd
        for batch_i, locations in enumerate(unit_locations):
            original_output[
                batch_i, locations, start_index:end_index
            ] = intervened_representation[batch_i] # [num_int, d]

    elif alignable_representation_type in {
        "head_query_output", "head_key_output", "head_value_output"}:
        # replacing [b, s, 3*d] with [b, num_int, s, dh]
        qkv_start_index = CONST_QKV_INDICES[alignable_representation_type]*n_embd
        for batch_i, locations in enumerate(unit_locations):
            for loc_i, loc in enumerate(locations):
                h_start_index = qkv_start_index+loc*attn_head_size
                h_end_index = qkv_start_index+(loc+1)*attn_head_size
                original_output[
                    batch_i, :, h_start_index:h_end_index
                ] = intervened_representation[batch_i, loc_i] # [s, dh]

    elif alignable_representation_type in {"head_attention_value_output"}:
        # replacing [b, s, d] with [b, num_int, s, dh]
        for batch_i, locations in enumerate(unit_locations):
            for loc_i, loc in enumerate(locations):
                h_start_index = loc*attn_head_size
                h_end_index = (loc+1)*attn_head_size
                original_output[
                    batch_i, :, h_start_index:h_end_index
                ] = intervened_representation[batch_i, loc_i] # [s, dh]
    else:
        # replacing [b, s, mlp_d/d] with [b, num_int, mlp_d/d]
        for batch_i, locations in enumerate(unit_locations):
            original_output[
                batch_i, locations
            ] = intervened_representation[batch_i]

