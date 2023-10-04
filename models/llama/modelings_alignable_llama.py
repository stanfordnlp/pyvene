"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.

We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.
"""


from models.constants import CONST_INPUT_HOOK, CONST_OUTPUT_HOOK


llama_type_to_module_mapping = {
}


llama_type_to_dimension_mapping = {
}