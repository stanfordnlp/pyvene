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


"""gpt_neox base model"""
gpt_neox_type_to_module_mapping = {
    'block_input': ("layers[%s]", CONST_INPUT_HOOK), 
    'block_output': ("layers[%s]", CONST_OUTPUT_HOOK), 
    'mlp_activation': ("layers[%s].mlp.act", CONST_OUTPUT_HOOK), 
    'mlp_output': ("layers[%s].mlp", CONST_OUTPUT_HOOK), 
    'mlp_input': ("layers[%s].mlp", CONST_INPUT_HOOK), 
    'attention_value_output': ("layers[%s].attention.dense", CONST_INPUT_HOOK),
    'head_attention_value_output': ("layers[%s].attention.dense", CONST_INPUT_HOOK),
    'attention_output': ("layers[%s].attention", CONST_OUTPUT_HOOK),
    'attention_input': ("layers[%s].attention", CONST_INPUT_HOOK),
    # 'query_output': ("layers[%s].attention.query_key_value", CONST_OUTPUT_HOOK),
    # 'key_output': ("layers[%s].attention.query_key_value", CONST_OUTPUT_HOOK),
    # 'value_output': ("layers[%s].attention.query_key_value", CONST_OUTPUT_HOOK),
    # 'head_query_output': ("layers[%s].attention.query_key_value", CONST_OUTPUT_HOOK),
    # 'head_key_output': ("layers[%s].attention.query_key_value", CONST_OUTPUT_HOOK),
    # 'head_value_output': ("layers[%s].attention.query_key_value", CONST_OUTPUT_HOOK),
}


gpt_neox_type_to_dimension_mapping = {
    'block_input': ("config.hidden_size", ), 
    'block_output': ("config.hidden_size", ), 
    'mlp_activation': ("config.intermediate_size", "config.hidden_size*4", ), 
    'mlp_output': ("config.hidden_size", ), 
    'mlp_input': ("config.hidden_size", ), 
    'attention_value_output': ("config.hidden_size", ),
    'head_attention_value_output': ("config.hidden_size/config.num_heads", ),
    'attention_output': ("config.hidden_size", ),
    'attention_input': ("config.hidden_size", ),
    # 'query_output': ("config.hidden_size", ),
    # 'key_output': ("config.hidden_size", ),
    # 'value_output': ("config.hidden_size", ),
    # 'head_query_output': ("config.hidden_size/config.num_heads", ),
    # 'head_key_output': ("config.hidden_size/config.num_heads", ),
    # 'head_value_output': ("config.hidden_size/config.num_heads", ),
}


"""gpt_neox model with LM head"""
gpt_neox_lm_type_to_module_mapping = {}
for k, v in gpt_neox_type_to_module_mapping.items():
    gpt_neox_lm_type_to_module_mapping[k] = (f"gpt_neox.{v[0]}", v[1])


gpt_neox_lm_type_to_dimension_mapping = gpt_neox_type_to_dimension_mapping


def create_gpt_neox(name="EleutherAI/pythia-70m", cache_dir="../../.huggingface_cache"):
    """Creates a GPT2 model, config, and tokenizer from the given name and revision"""
    from transformers import GPTNeoXForCausalLM, AutoTokenizer, GPTNeoXConfig
    
    config = GPTNeoXConfig.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    gpt_neox = GPTNeoXForCausalLM.from_pretrained(name)
    print("loaded model")
    return config, tokenizer, gpt_neox


def split_heads(tensor, num_heads, attn_head_size):
    """
    Splits hidden_size dim into attn_head_size and num_heads
    """
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(new_shape)
    return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)


# def split_heads_qkv(tensor, num_heads, attn_head_size, return_type):
#     """
#     Splits hidden_size dim into attn_head_size, 3, and num_heads
#     """
#     new_shape = tensor.size()[:-1] + (num_heads, 3, attn_head_size)
#     tensor = tensor.view(new_shape) # (batch, seq_length, num_heads, q/k/v, head_size)
#     return tensor.permute(3, 0, 2, 1, 4) # (q/k/v, batch, head, seq_length, head_features)

def gpt_neox_output_to_subcomponent(
    output, alignable_representation_type, model_config
):
    n_embd = model_config.hidden_size
    num_heads = model_config.num_attention_heads
    attn_head_size = n_embd // num_heads
    # if alignable_representation_type in {
    #     "head_query_output", "head_key_output", "head_value_output"
    # }:
    #     return split_heads_qkv(output, num_heads, attn_head_size)  
    if alignable_representation_type == "head_attention_value_output":
        return split_heads(output, num_heads, attn_head_size)
    else:
        return output


def gpt_neox_scatter_intervention_output(
    original_output, intervened_representation,
    alignable_representation_type,
    unit_locations, model_config
):
    assert original_output.shape[0] == \
            unit_locations.shape[0]
    assert intervened_representation.shape[0] == \
            unit_locations.shape[0]
    
    n_embd = model_config.hidden_size
    num_heads = model_config.num_attention_heads
    attn_head_size = n_embd // num_heads
    
    # if alignable_representation_type in {
    #     "head_query_output", "head_key_output", "head_value_output"}:
    #     pass
    if alignable_representation_type in {"head_attention_value_output"}:
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

