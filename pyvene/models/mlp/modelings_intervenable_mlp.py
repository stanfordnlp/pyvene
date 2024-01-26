"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.

We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.
"""


from ..constants import *


"""mlp base model"""
mlp_type_to_module_mapping = {
    "block_input": ("h[%s]", CONST_INPUT_HOOK),
    "block_output": ("h[%s]", CONST_OUTPUT_HOOK),
    "mlp_activation": ("h[%s].act", CONST_OUTPUT_HOOK),
}


mlp_type_to_dimension_mapping = {
    "block_input": ("h_dim",),
    "block_output": ("h_dim",),
    "mlp_activation": ("h_dim",),
}


"""mlp model with classification head"""
mlp_classifier_type_to_module_mapping = {}
for k, v in mlp_type_to_module_mapping.items():
    mlp_classifier_type_to_module_mapping[k] = (f"mlp.{v[0]}", v[1])

mlp_classifier_type_to_dimension_mapping = mlp_type_to_dimension_mapping


def create_mlp_classifier(
    config, tokenizer_name=None, cache_dir=None
):
    """Creates a MLP model, config, and tokenizer from the given name and revision"""
    from transformers import AutoTokenizer
    from pyvene.models.mlp.modelings_mlp import MLPForClassification

    tokenizer = None
    if tokenizer_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    mlp = MLPForClassification(config=config)
    print("loaded model")
    return config, tokenizer, mlp
