"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.

We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.
"""


from ..constants import *


"""electra base model"""
electra_type_to_module_mapping = {
    "block_input": ("encoder.layer[%s]", CONST_INPUT_HOOK),
    "block_output": ("encoder.layer[%s]", CONST_OUTPUT_HOOK),
    "mlp_activation": ("encoder.layer[%s].intermediate", CONST_OUTPUT_HOOK),
    "mlp_output": ("encoder.layer[%s].output", CONST_OUTPUT_HOOK),
    "mlp_input": ("encoder.layer[%s].intermediate", CONST_INPUT_HOOK),
    "attention_value_output": ("encoder.layer[%s].attention.output.dense", CONST_INPUT_HOOK),
    "head_attention_value_output": ("encoder.layer[%s].attention.output.dense", CONST_INPUT_HOOK, (split_head_and_permute, "num_attention_heads")),
    "attention_output": ("encoder.layer[%s].attention.output", CONST_OUTPUT_HOOK),
    "attention_input": ("encoder.layer[%s].attention", CONST_INPUT_HOOK),
    "query_output": ("encoder.layer[%s].attention.self.query", CONST_OUTPUT_HOOK),
    "key_output": ("encoder.layer[%s].attention.self.key", CONST_OUTPUT_HOOK),
    "value_output": ("encoder.layer[%s].attention.self.value", CONST_OUTPUT_HOOK),
    "head_query_output": ("encoder.layer[%s].attention.self.query", CONST_OUTPUT_HOOK, (split_head_and_permute, "num_attention_heads")),
    "head_key_output": ("encoder.layer[%s].attention.self.key", CONST_OUTPUT_HOOK, (split_head_and_permute, "num_attention_heads")),
    "head_value_output": ("encoder.layer[%s].attention.self.value", CONST_OUTPUT_HOOK, (split_head_and_permute, "num_attention_heads")),
}


electra_type_to_dimension_mapping = {
    "num_attention_heads": ("num_attention_heads",),
    "block_input": ("hidden_size",),
    "block_output": ("hidden_size",),
    "mlp_activation": ("intermediate_size",),
    "mlp_output": ("hidden_size",),
    "mlp_input": ("hidden_size",),
    "attention_value_output": ("hidden_size",),
    "head_attention_value_output": ("hidden_size/num_attention_heads",),
    "attention_output": ("hidden_size",),
    "attention_input": ("hidden_size",),
    "query_output": ("hidden_size",),
    "key_output": ("hidden_size",),
    "value_output": ("hidden_size",),
    "head_query_output": ("hidden_size/num_attention_heads",),
    "head_key_output": ("hidden_size/num_attention_heads",),
    "head_value_output": ("hidden_size/num_attention_heads",),
}


"""electra model with masked LM head"""
electra_mlm_type_to_module_mapping = {}
for k, v in electra_type_to_module_mapping.items():
    electra_mlm_type_to_module_mapping[k] = (f"electra.{v[0]}", ) + v[1:]

electra_mlm_type_to_dimension_mapping = electra_type_to_dimension_mapping


"""electra model with classifier head"""
electra_classifier_type_to_module_mapping = {}
for k, v in electra_type_to_module_mapping.items():
    electra_classifier_type_to_module_mapping[k] = (f"electra.{v[0]}", ) + v[1:]

electra_classifier_type_to_dimension_mapping = electra_type_to_dimension_mapping


def create_electra(name="google/electra-base-discriminator", cache_dir=None):
    """Creates an ELECTRA base model, config, and tokenizer from the given name"""
    from transformers import ElectraModel, ElectraTokenizer, ElectraConfig

    config = ElectraConfig.from_pretrained(name)
    tokenizer = ElectraTokenizer.from_pretrained(name)
    electra = ElectraModel.from_pretrained(name, config=config, cache_dir=cache_dir)
    print("loaded model")
    return config, tokenizer, electra


def create_electra_mlm(name="google/electra-base-generator", config=None, cache_dir=None):
    """Creates an ElectraForMaskedLM, config, and tokenizer from the given name"""
    from transformers import ElectraForMaskedLM, ElectraTokenizer, ElectraConfig

    if config is None:
        config = ElectraConfig.from_pretrained(name)
        tokenizer = ElectraTokenizer.from_pretrained(name)
        electra = ElectraForMaskedLM.from_pretrained(name, config=config, cache_dir=cache_dir)
    else:
        tokenizer = None
        electra = ElectraForMaskedLM(config=config)
    print("loaded model")
    return config, tokenizer, electra


def create_electra_classifier(name="google/electra-base-discriminator", config=None, cache_dir=None):
    """Creates an ElectraForSequenceClassification, config, and tokenizer from the given name"""
    from transformers import ElectraForSequenceClassification, ElectraTokenizer, ElectraConfig

    if config is None:
        config = ElectraConfig.from_pretrained(name)
        tokenizer = ElectraTokenizer.from_pretrained(name)
        electra = ElectraForSequenceClassification.from_pretrained(name, config=config, cache_dir=cache_dir)
    else:
        tokenizer = None
        electra = ElectraForSequenceClassification(config=config)
    print("loaded model")
    return config, tokenizer, electra
