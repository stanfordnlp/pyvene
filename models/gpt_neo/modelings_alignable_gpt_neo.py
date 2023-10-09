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


"""gpt_neo base model"""
gpt_neo_type_to_module_mapping = {
    'block_input': ("h[%s]", CONST_INPUT_HOOK), 
    'block_output': ("h[%s]", CONST_OUTPUT_HOOK), 
    'mlp_activation': ("h[%s].mlp.act", CONST_OUTPUT_HOOK), 
    'mlp_output': ("h[%s].mlp", CONST_OUTPUT_HOOK), 
    'mlp_input': ("h[%s].mlp", CONST_INPUT_HOOK), 
    'attention_value_output': ("h[%s].attn.out_proj", CONST_INPUT_HOOK),
    'head_attention_value_output': ("h[%s].attn.out_proj", CONST_INPUT_HOOK),
    'attention_output': ("h[%s].attn", CONST_OUTPUT_HOOK),
    'attention_input': ("h[%s].attn", CONST_INPUT_HOOK),
    'query_output': ("h[%s].attn.q_proj", CONST_OUTPUT_HOOK),
    'key_output': ("h[%s].attn.k_proj", CONST_OUTPUT_HOOK),
    'value_output': ("h[%s].attn.v_proj", CONST_OUTPUT_HOOK),
    'head_query_output': ("h[%s].attn.q_proj", CONST_OUTPUT_HOOK),
    'head_key_output': ("h[%s].attn.k_proj", CONST_OUTPUT_HOOK),
    'head_value_output': ("h[%s].attn.v_proj", CONST_OUTPUT_HOOK),
}


gpt_neo_type_to_dimension_mapping = {
    'block_input': ("config.hidden_size", ), 
    'block_output': ("config.hidden_size", ), 
    'mlp_activation': ("config.intermediate_size", "config.hidden_size*4", ), 
    'mlp_output': ("config.hidden_size", ), 
    'mlp_input': ("config.hidden_size", ), 
    'attention_value_output': ("config.hidden_size", ),
    'head_attention_value_output': ("config.hidden_size/config.num_heads", ),
    'attention_output': ("config.hidden_size", ),
    'attention_input': ("config.hidden_size", ),
    'query_output': ("config.hidden_size", ),
    'key_output': ("config.hidden_size", ),
    'value_output': ("config.hidden_size", ),
    'head_query_output': ("config.hidden_size/config.num_heads", ),
    'head_key_output': ("config.hidden_size/config.num_heads", ),
    'head_value_output': ("config.hidden_size/config.num_heads", ),
}


"""gpt_neo model with LM head"""
gpt_neo_lm_type_to_module_mapping = {}
for k, v in gpt_neo_type_to_module_mapping.items():
    gpt_neo_lm_type_to_module_mapping[k] = (f"transformer.{v[0]}", v[1])


gpt_neo_lm_type_to_dimension_mapping = gpt_neo_type_to_dimension_mapping


def create_gpt_neo(name="roneneldan/TinyStories-33M", cache_dir="../../.huggingface_cache"):
    """Creates a GPT2 model, config, and tokenizer from the given name and revision"""
    from transformers import GPTNeoForCausalLM, GPT2Tokenizer, GPTNeoConfig
    
    config = GPTNeoConfig.from_pretrained(name)
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M") # not sure
    gpt_neo = GPTNeoForCausalLM.from_pretrained(name)
    print("loaded model")
    return config, tokenizer, gpt_neo


def split_heads(tensor, num_heads, attn_head_size):
    """
    Splits hidden_size dim into attn_head_size and num_heads
    """
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(new_shape)
    return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)


def gpt_neo_output_to_subcomponent(
    output, alignable_representation_type, model_config
):
    n_embd = model_config.hidden_size
    num_heads = model_config.num_attention_heads
    attn_head_size = n_embd // num_heads
    if alignable_representation_type in {
        "head_query_output", "head_key_output", "head_value_output",
        "head_attention_value_output",
    }:
        return split_heads(output, num_heads, attn_head_size)
    else:
        return output


def gpt_neo_scatter_intervention_output(
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
    
    if alignable_representation_type in {
        "head_query_output", "head_key_output", "head_value_output",
        "head_attention_value_output"}:
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

