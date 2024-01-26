"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.

We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.
"""


from ..constants import *


"""gpt2 base model"""
backpack_gpt2_lm_type_to_module_mapping = {
    "sense_network_output": ("backpack.sense_network", CONST_OUTPUT_HOOK),
}


backpack_gpt2_lm_type_to_dimension_mapping = {
    "sense_network_output": ("n_embd",),
}

def create_backpack_gpt2(name="stanfordnlp/backpack-gpt2", cache_dir=None):
    """Creates a GPT2 model, config, and tokenizer from the given name and revision"""
    # Load model directly
    from transformers import AutoTokenizer
    from pyvene.models.backpack_gpt2.modelings_backpack_gpt2 import BackpackGPT2LMHeadModel

    tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
    model = BackpackGPT2LMHeadModel.from_pretrained(name, trust_remote_code=True)
    print("loaded model")
    return model.config, tokenizer, model

