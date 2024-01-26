"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.

We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.
"""


from ..constants import *


"""gru base model"""
gru_type_to_module_mapping = {
    "cell_input": ("cells[%s]", CONST_INPUT_HOOK),
    "reset_gate_input": ("cells[%s].reset_act", CONST_INPUT_HOOK),
    "update_gate_input": ("cells[%s].update_act", CONST_INPUT_HOOK),
    "new_gate_input": ("cells[%s].new_act", CONST_INPUT_HOOK),
    "reset_gate_output": ("cells[%s].reset_act", CONST_OUTPUT_HOOK),
    "update_gate_output": ("cells[%s].update_act", CONST_OUTPUT_HOOK),
    "new_gate_output": ("cells[%s].new_act", CONST_OUTPUT_HOOK),
    "x2h_output": ("cells[%s].x2h", CONST_OUTPUT_HOOK),
    "h2h_output": ("cells[%s].h2h", CONST_OUTPUT_HOOK),
    "reset_x2h_output": ("cells[%s].x2h", CONST_OUTPUT_HOOK, (split_three, 0)),
    "update_x2h_output": ("cells[%s].x2h", CONST_OUTPUT_HOOK, (split_three, 1)),
    "new_x2h_output": ("cells[%s].x2h", CONST_OUTPUT_HOOK, (split_three, 2)),
    "reset_h2h_output": ("cells[%s].h2h", CONST_OUTPUT_HOOK, (split_three, 0)),
    "update_h2h_output": ("cells[%s].h2h", CONST_OUTPUT_HOOK, (split_three, 1)),
    "new_h2h_output": ("cells[%s].h2h", CONST_OUTPUT_HOOK, (split_three, 2)),
    "cell_output": ("cells[%s]", CONST_OUTPUT_HOOK),
}


gru_type_to_dimension_mapping = {
    "cell_input": ("h_dim",),
    "reset_gate_input": ("h_dim",),
    "update_gate_input": ("h_dim",),
    "new_gate_input": ("h_dim",),
    "reset_gate_output": ("h_dim",),
    "update_gate_output": ("h_dim",),
    "new_gate_output": ("h_dim",),
    "x2h_output": ("h_dim*3",),
    "h2h_output": ("h_dim*3",),
    "reset_x2h_output": ("h_dim",),
    "update_x2h_output": ("h_dim",),
    "new_x2h_output": ("h_dim",),
    "reset_h2h_output": ("h_dim",),
    "update_h2h_output": ("h_dim",),
    "new_h2h_output": ("h_dim",),
    "cell_output": ("h_dim",),
}


"""mlp model with classification head"""
gru_classifier_type_to_module_mapping = {}
for k, v in gru_type_to_module_mapping.items():
    gru_classifier_type_to_module_mapping[k] = (f"gru.{v[0]}", ) + v[1:]

gru_classifier_type_to_dimension_mapping = gru_type_to_dimension_mapping


"""mlp model with lm head"""
gru_lm_type_to_module_mapping = {}
for k, v in gru_type_to_module_mapping.items():
    gru_lm_type_to_module_mapping[k] = (f"gru.{v[0]}", v[1])

gru_lm_type_to_dimension_mapping = gru_type_to_dimension_mapping


def create_gru(config, tokenizer_name=None, cache_dir=None):
    """Creates a GRU model, config, and tokenizer from the given name and revision"""
    from transformers import AutoTokenizer
    from models.gru.modelings_gru import GRUModel

    tokenizer = None
    if tokenizer_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    mlp = GRUModel(config=config)
    print("loaded model")
    return config, tokenizer, mlp


def create_gru_lm(config, tokenizer_name=None, cache_dir=None):
    """Creates a GRU model, config, and tokenizer from the given name and revision"""
    from transformers import AutoTokenizer
    from models.gru.modelings_gru import GRULMHeadModel

    tokenizer = None
    if tokenizer_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    mlp = GRULMHeadModel(config=config)
    print("loaded model")
    return config, tokenizer, mlp


def create_gru_classifier(
    config, tokenizer_name=None, cache_dir=None
):
    """Creates a GRU model, config, and tokenizer from the given name and revision"""
    from transformers import AutoTokenizer
    from pyvene.models.gru.modelings_gru import GRUForClassification

    tokenizer = None
    if tokenizer_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    mlp = GRUForClassification(config=config)
    print("loaded model")
    return config, tokenizer, mlp
