"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.

We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.
"""


from ..constants import *

"""blip base model"""
blip_type_to_module_mapping = {
    # 'vis.block_input': ("vision_model.encoder.layers[%s]", CONST_INPUT_HOOK),
    # 'vis.block_output': ("vision_model.encoder.layers[%s]", CONST_OUTPUT_HOOK),
    # 'vis.mlp_activation': ("vision_model.encoder.layers[%s].mlp.fc1", CONST_OUTPUT_HOOK),
    # 'vis.mlp_output': ("vision_model.encoder.layers[%s].mlp", CONST_OUTPUT_HOOK),
    # 'vis.mlp_input': ("vision_model.encoder.layers[%s].mlp", CONST_INPUT_HOOK),
    # 'vis.attention_value_output': ("vision_model.encoder.layers[%s].self_attn.projection", CONST_INPUT_HOOK),
    # 'vis.attention_output': ("vision_model.encoder.layers[%s].self_attn", CONST_OUTPUT_HOOK),
    # 'vis.attention_input': ("vision_model.encoder.layers[%s].self_attn", CONST_INPUT_HOOK),
    "block_input": ("text_encoder.encoder.layer[%s]", CONST_INPUT_HOOK),
    "block_output": ("text_encoder.encoder.layer[%s]", CONST_INPUT_HOOK),
    "mlp_activation": (
        "text_encoder.encoder.layer[%s].intermediate.dense",
        CONST_OUTPUT_HOOK,
    ),
    "mlp_output": ("text_encoder.encoder.layer[%s].output", CONST_OUTPUT_HOOK),
    "mlp_input": ("text_encoder.encoder.layer[%s].intermediate", CONST_INPUT_HOOK),
    "attention_value_output": (
        "text_encoder.encoder.layer[%s].attention.output.dense",
        CONST_INPUT_HOOK,
    ),
    "attention_output": (
        "text_encoder.encoder.layer[%s].attention.output",
        CONST_OUTPUT_HOOK,
    ),
    "attention_input": ("text_encoder.encoder.layer[%s].attention", CONST_INPUT_HOOK),
    # 'block_input': ("text_decoder.bert.encoder.layer[%s]", CONST_INPUT_HOOK),
    # 'block_output': ("text_decoder.bert.encoder.layer[%s]", CONST_INPUT_HOOK),
    # 'mlp_activation': ("text_decoder.bert.encoder.layer[%s].intermediate.dense", CONST_OUTPUT_HOOK),
    # 'mlp_output': ("text_decoder.bert.encoder.layer[%s].output", CONST_OUTPUT_HOOK),
    # 'mlp_input': ("text_decoder.bert.encoder.layer[%s].intermediate", CONST_INPUT_HOOK),
    # 'attention_value_output': ("text_decoder.bert.encoder.layer[%s].attention.output.dense", CONST_INPUT_HOOK),
    # 'attention_output': ("text_decoder.bert.encoder.layer[%s].attention.output", CONST_OUTPUT_HOOK),
    # 'attention_input': ("text_decoder.bert.encoder.layer[%s].attention", CONST_INPUT_HOOK),
    # 'cross_attention_value_output': ("text_decoder.bert.encoder.layer[%s].crossattention.output.dense", CONST_INPUT_HOOK),
    # 'cross_attention_output': ("text_decoder.bert.encoder.layer[%s].crossattention.output", CONST_OUTPUT_HOOK),
    # 'cross_attention_input': ("text_decoder.bert.encoder.layer[%s].crossattention", CONST_INPUT_HOOK),
}


blip_type_to_dimension_mapping = {
    # 'vis.block_input': ("image_text_hidden_size", ),
    # 'vis.block_output': ("image_text_hidden_size", ),
    # 'vis.mlp_activation': ("projection_dim", ),
    # 'vis.mlp_output': ("image_text_hidden_size", ),
    # 'vis.mlp_input': ("image_text_hidden_size", ),
    # 'vis.attention_value_output': ("image_text_hidden_size/text_config.num_attention_heads", ),
    # 'vis.attention_output': ("image_text_hidden_size", ),
    # 'vis.attention_input': ("image_text_hidden_size", ),
    # 'lang.block_input': ("image_text_hidden_size", ),
    # 'lang.block_output': ("image_text_hidden_size", ),
    # 'lang.mlp_activation': ("projection_dim", ),
    # 'lang.mlp_output': ("image_text_hidden_size", ),
    # 'lang.mlp_input': ("image_text_hidden_size", ),
    # 'lang.attention_value_output': ("image_text_hidden_size/text_config.num_attention_heads", ),
    # 'lang.attention_output': ("image_text_hidden_size", ),
    # 'lang.attention_input': ("image_text_hidden_size", ),
    "block_input": ("image_text_hidden_size",),
    "block_output": ("image_text_hidden_size",),
    "mlp_activation": ("projection_dim",),
    "mlp_output": ("image_text_hidden_size",),
    "mlp_input": ("image_text_hidden_size",),
    "attention_value_output": (
        "image_text_hidden_size/text_config.num_attention_heads",
    ),
    "attention_output": ("image_text_hidden_size",),
    "attention_input": ("image_text_hidden_size",),
    "cross_attention_value_output": (
        "image_text_hidden_size/text_config.num_attention_heads",
    ),
    "cross_attention_output": ("image_text_hidden_size",),
    "cross_attention_input": ("image_text_hidden_size",),
}


"""blip model with wrapper"""
blip_wrapper_type_to_module_mapping = {}
for k, v in blip_type_to_module_mapping.items():
    blip_wrapper_type_to_module_mapping[k] = (
        v[0].replace("text_encoder", "model_text_enc"),
        v[1],
    )


blip_wrapper_type_to_dimension_mapping = blip_type_to_dimension_mapping


def create_blip(name="Salesforce/blip-vqa-base", cache_dir=None):
    """Creates a BLIP VQA model, config, and tokenizer from the given name and revision"""
    from transformers import BlipConfig, BlipProcessor, BlipForQuestionAnswering

    config = BlipConfig.from_pretrained(name)
    processor = BlipProcessor.from_pretrained(name)
    blip = BlipForQuestionAnswering.from_pretrained(
        name, config=config, cache_dir=cache_dir
    )
    print("loaded model")
    return config, processor, blip
