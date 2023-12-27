"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.

We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.
"""


from models.constants import CONST_INPUT_HOOK, CONST_OUTPUT_HOOK, CONST_QKV_INDICES

"""blip base model"""
blip_type_to_module_mapping = {
    'vis.block_input': ("vision_model.encoder.layers[%s]", CONST_INPUT_HOOK),
    'vis.block_output': ("vision_model.encoder.layers[%s]", CONST_OUTPUT_HOOK),
    'vis.mlp_activation': ("vision_model.encoder.layers[%s].mlp.fc1", CONST_OUTPUT_HOOK),
    'vis.mlp_output': ("vision_model.encoder.layers[%s].mlp", CONST_OUTPUT_HOOK),
    'vis.mlp_input': ("vision_model.encoder.layers[%s].mlp", CONST_INPUT_HOOK),
    'vis.attention_value_output': ("vision_model.encoder.layers[%s].self_attn.projection", CONST_INPUT_HOOK),
    'vis.attention_output': ("vision_model.encoder.layers[%s].self_attn", CONST_OUTPUT_HOOK),
    'vis.attention_input': ("vision_model.encoder.layers[%s].self_attn", CONST_INPUT_HOOK),

    'lang.block_input': ("text_encoder.encoder.layer[%s]", CONST_INPUT_HOOK),
    'lang.block_output': ("text_encoder.encoder.layer[%s]", CONST_INPUT_HOOK),
    'lang.mlp_activation': ("text_encoder.encoder.layer[%s].intermediate.dense", CONST_OUTPUT_HOOK),
    'lang.mlp_output': ("text_encoder.encoder.layer[%s].output", CONST_OUTPUT_HOOK),
    'lang.mlp_input': ("text_encoder.encoder.layer[%s].intermediate", CONST_INPUT_HOOK),
    'lang.attention_value_output': ("text_encoder.encoder.layer[%s].attention.output.dense", CONST_INPUT_HOOK),
    'lang.attention_output': ("text_encoder.encoder.layer[%s].attention.output", CONST_OUTPUT_HOOK),
    'lang.attention_input': ("text_encoder.encoder.layer[%s].attention", CONST_INPUT_HOOK),

    'decoder.block_input': ("text_decoder.encoder.layers[%s]", CONST_INPUT_HOOK),
    'decoder.block_output': ("text_decoder.encoder.layer[%s]", CONST_INPUT_HOOK),
    'decoder.mlp_activation': ("text_decoder.encoder.layer[%s].intermediate.dense", CONST_OUTPUT_HOOK),
    'decoder.mlp_output': ("text_decoder.encoder.layer[%s].output", CONST_OUTPUT_HOOK),
    'decoder.mlp_input': ("text_decoder.encoder.layer[%s].intermediate", CONST_INPUT_HOOK),
    'decoder.self_attention_value_output': ("text_decoder.encoder.layer[%s].attention.output.dense", CONST_INPUT_HOOK),
    'decoder.self_attention_output': ("text_decoder.encoder.layer[%s].attention.output", CONST_OUTPUT_HOOK),
    'decoder.self_attention_input': ("text_decoder.encoder.layer[%s].attention", CONST_INPUT_HOOK),
    'decoder.cross_attention_value_output': ("text_decoder.encoder.layer[%s].crossattention.output.dense", CONST_INPUT_HOOK),
    'decoder.cross_attention_output': ("text_decoder.encoder.layer[%s].crossattention.output", CONST_OUTPUT_HOOK),
    'decoder.cross_attention_input': ("text_decoder.encoder.layer[%s].crossattention", CONST_INPUT_HOOK),
}


blip_type_to_dimension_mapping = {
    'vis.block_input': ("image_text_hidden_size", ), 
    'vis.block_output': ("image_text_hidden_size", ), 
    'vis.mlp_activation': ("projection_dim", ),
    'vis.mlp_output': ("image_text_hidden_size", ),
    'vis.mlp_input': ("image_text_hidden_size", ),
    'vis.attention_value_output': ("image_text_hidden_size/text_config.num_attention_heads", ),
    'vis.attention_output': ("image_text_hidden_size", ),
    'vis.attention_input': ("image_text_hidden_size", ),

    'lang.block_input': ("image_text_hidden_size", ),
    'lang.block_output': ("image_text_hidden_size", ),
    'lang.mlp_activation': ("projection_dim", ),
    'lang.mlp_output': ("image_text_hidden_size", ),
    'lang.mlp_input': ("image_text_hidden_size", ),
    'lang.attention_value_output': ("image_text_hidden_size/text_config.num_attention_heads", ),
    'lang.attention_output': ("image_text_hidden_size", ),
    'lang.attention_input': ("image_text_hidden_size", ),

    'decoder.block_input': ("image_text_hidden_size", ),
    'decoder.block_output': ("image_text_hidden_size", ),
    'decoder.mlp_activation': ("projection_dim", ),
    'decoder.mlp_output': ("image_text_hidden_size", ),
    'decoder.mlp_input': ("image_text_hidden_size", ),
    'decoder.self_attention_value_output': (),
    'decoder.self_attention_output': ("image_text_hidden_size", ),
    'decoder.self_attention_input': ("image_text_hidden_size", ),
    'decoder.cross_attention_value_output': ("image_text_hidden_size/text_config.num_attention_heads", ),
    'decoder.cross_attention_output': ("image_text_hidden_size", ),
    'decoder.cross_attention_input': ("image_text_hidden_size", ),
}

blip_lm_type_to_dimension_mapping = blip_type_to_dimension_mapping

def create_blip(name="Salesforce/blip-vqa-base", cache_dir="../../.huggingface_cache"):
    """Creates a GPT2 model, config, and tokenizer from the given name and revision"""
    from transformers import BlipConfig, BlipProcessor, BlipForQuestionAnswering

    config = BlipConfig.from_pretrained(name)
    processor = BlipProcessor.from_pretrained(name)
    blip = BlipForQuestionAnswering.from_pretrained(name, config=config, cache_dir=cache_dir)
    print("loaded model")
    return config, processor, blip