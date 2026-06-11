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


"""t5 base model (encoder-only anchor paths)

T5 is an encoder-decoder model. For parity with the existing whisper
mapping in this library, the standard anchor names address the encoder
stack. The encoder block layout is:
    encoder.block[i].layer[0]  -> self-attention sublayer
    encoder.block[i].layer[1]  -> feed-forward sublayer

Note on dimensions: T5 attention uses an inner dimension of
``num_heads * d_kv`` (which can differ from ``d_model``), so the q/k/v/o
projections are sized by ``inner_dim``, not by ``d_model``. We therefore
expose ``inner_dim = num_heads*d_kv`` via the dimension mapping.
"""
t5_type_to_module_mapping = {
    "block_input": ("encoder.block[%s]", CONST_INPUT_HOOK),
    "block_output": ("encoder.block[%s]", CONST_OUTPUT_HOOK),
    "mlp_activation": ("encoder.block[%s].layer[1].DenseReluDense.act", CONST_OUTPUT_HOOK),
    "mlp_output": ("encoder.block[%s].layer[1].DenseReluDense.wo", CONST_OUTPUT_HOOK),
    "mlp_input": ("encoder.block[%s].layer[1].DenseReluDense.wi", CONST_INPUT_HOOK),
    "attention_value_output": ("encoder.block[%s].layer[0].SelfAttention.o", CONST_INPUT_HOOK),
    "head_attention_value_output": ("encoder.block[%s].layer[0].SelfAttention.o", CONST_INPUT_HOOK, (split_head_and_permute, "num_heads")),
    "attention_output": ("encoder.block[%s].layer[0].SelfAttention", CONST_OUTPUT_HOOK),
    "attention_input": ("encoder.block[%s].layer[0].SelfAttention", CONST_INPUT_HOOK),
    "query_output": ("encoder.block[%s].layer[0].SelfAttention.q", CONST_OUTPUT_HOOK),
    "key_output": ("encoder.block[%s].layer[0].SelfAttention.k", CONST_OUTPUT_HOOK),
    "value_output": ("encoder.block[%s].layer[0].SelfAttention.v", CONST_OUTPUT_HOOK),
    "head_query_output": ("encoder.block[%s].layer[0].SelfAttention.q", CONST_OUTPUT_HOOK, (split_head_and_permute, "num_heads")),
    "head_key_output": ("encoder.block[%s].layer[0].SelfAttention.k", CONST_OUTPUT_HOOK, (split_head_and_permute, "num_heads")),
    "head_value_output": ("encoder.block[%s].layer[0].SelfAttention.v", CONST_OUTPUT_HOOK, (split_head_and_permute, "num_heads")),
}


t5_type_to_dimension_mapping = {
    "num_heads": ("num_heads",),
    "block_input": ("d_model",),
    "block_output": ("d_model",),
    "mlp_activation": ("d_ff",),
    "mlp_output": ("d_model",),
    "mlp_input": ("d_model",),
    "attention_value_output": ("num_heads*d_kv",),
    "head_attention_value_output": ("d_kv",),
    "attention_output": ("d_model",),
    "attention_input": ("d_model",),
    "query_output": ("num_heads*d_kv",),
    "key_output": ("num_heads*d_kv",),
    "value_output": ("num_heads*d_kv",),
    "head_query_output": ("d_kv",),
    "head_key_output": ("d_kv",),
    "head_value_output": ("d_kv",),
}


"""t5 encoder-only model (T5EncoderModel)

T5EncoderModel exposes the encoder stack at the top level (i.e. its
module names are already ``encoder.block[i]....``), so the encoder-only
anchor paths above can be reused as-is.
"""
t5_encoder_type_to_module_mapping = dict(t5_type_to_module_mapping)
t5_encoder_type_to_dimension_mapping = t5_type_to_dimension_mapping


"""t5 model with conditional generation head (T5ForConditionalGeneration)

T5ForConditionalGeneration wraps the encoder/decoder under no additional
prefix — the encoder is still reachable at ``encoder.block[i]...`` — so
the encoder-only anchor mapping can be reused unchanged.
"""
t5_lm_type_to_module_mapping = dict(t5_type_to_module_mapping)
t5_lm_type_to_dimension_mapping = t5_type_to_dimension_mapping


def create_t5(name="google-t5/t5-small", cache_dir=None):
    """Creates a T5Model, config, and tokenizer from the given name."""
    from transformers import T5Config, T5Model, T5Tokenizer

    config = T5Config.from_pretrained(name)
    tokenizer = T5Tokenizer.from_pretrained(name)
    t5 = T5Model.from_pretrained(name, config=config, cache_dir=cache_dir)
    print("loaded model")
    return config, tokenizer, t5


def create_t5_lm(name="google-t5/t5-small", config=None, cache_dir=None):
    """Creates a T5ForConditionalGeneration, config, and tokenizer."""
    from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

    if config is None:
        config = T5Config.from_pretrained(name)
        tokenizer = T5Tokenizer.from_pretrained(name)
        t5 = T5ForConditionalGeneration.from_pretrained(
            name, config=config, cache_dir=cache_dir
        )
    else:
        tokenizer = None
        t5 = T5ForConditionalGeneration(config=config)
    print("loaded model")
    return config, tokenizer, t5


def create_t5_encoder(name="google-t5/t5-small", config=None, cache_dir=None):
    """Creates a T5EncoderModel, config, and tokenizer."""
    from transformers import T5Config, T5EncoderModel, T5Tokenizer

    if config is None:
        config = T5Config.from_pretrained(name)
        tokenizer = T5Tokenizer.from_pretrained(name)
        t5 = T5EncoderModel.from_pretrained(name, config=config, cache_dir=cache_dir)
    else:
        tokenizer = None
        t5 = T5EncoderModel(config=config)
    print("loaded model")
    return config, tokenizer, t5
