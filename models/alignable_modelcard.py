from models.constants import *
from models.llama.modelings_alignable_llama import *
from models.gpt2.modelings_alignable_gpt2 import *
from models.gpt_neo.modelings_alignable_gpt_neo import *
from models.gpt_neox.modelings_alignable_gpt_neox import *
from models.mlp.modelings_alignable_mlp import *
from models.gru.modelings_alignable_gru import *
from models.blip.modelings_alignable_blip import *


#########################################################################
"""
Below are functions that you need to modify if you add
a new model arch type in this library.

We put them in front so it is easier to keep track of
things that need to be changed.
"""

import transformers.models as hf_models
from models.blip.modelings_blip import BlipWrapper
from models.mlp.modelings_mlp import MLPModel, MLPForClassification
from models.gru.modelings_gru import GRUModel, GRULMHeadModel, GRUForClassification

global type_to_module_mapping
global type_to_dimension_mapping
global output_to_subcomponent_fn_mapping
global scatter_intervention_output_fn_mapping


type_to_module_mapping = {
    hf_models.gpt2.modeling_gpt2.GPT2Model: gpt2_type_to_module_mapping,
    hf_models.gpt2.modeling_gpt2.GPT2LMHeadModel: gpt2_lm_type_to_module_mapping,
    hf_models.llama.modeling_llama.LlamaModel: llama_type_to_module_mapping,
    hf_models.llama.modeling_llama.LlamaForCausalLM: llama_lm_type_to_module_mapping,
    hf_models.gpt_neo.modeling_gpt_neo.GPTNeoModel: gpt_neo_type_to_module_mapping,
    hf_models.gpt_neo.modeling_gpt_neo.GPTNeoForCausalLM: gpt_neo_lm_type_to_module_mapping,
    hf_models.gpt_neox.modeling_gpt_neox.GPTNeoXModel: gpt_neox_type_to_module_mapping,
    hf_models.gpt_neox.modeling_gpt_neox.GPTNeoXForCausalLM: gpt_neox_lm_type_to_module_mapping,
    hf_models.blip.modeling_blip.BlipForQuestionAnswering: blip_type_to_module_mapping,
    BlipWrapper: blip_wrapper_type_to_module_mapping,
    MLPModel: mlp_type_to_module_mapping,
    MLPForClassification: mlp_classifier_type_to_module_mapping,
    GRUModel: gru_type_to_module_mapping,
    GRULMHeadModel: gru_lm_type_to_module_mapping,
    GRUForClassification: gru_classifier_type_to_module_mapping,
    # new model type goes here after defining the model files
}


type_to_dimension_mapping = {
    hf_models.gpt2.modeling_gpt2.GPT2Model: gpt2_type_to_dimension_mapping,
    hf_models.gpt2.modeling_gpt2.GPT2LMHeadModel: gpt2_lm_type_to_dimension_mapping,
    hf_models.llama.modeling_llama.LlamaModel: llama_type_to_dimension_mapping,
    hf_models.llama.modeling_llama.LlamaForCausalLM: llama_lm_type_to_dimension_mapping,
    hf_models.gpt_neo.modeling_gpt_neo.GPTNeoModel: gpt_neo_type_to_dimension_mapping,
    hf_models.gpt_neo.modeling_gpt_neo.GPTNeoForCausalLM: gpt_neo_lm_type_to_dimension_mapping,
    hf_models.gpt_neox.modeling_gpt_neox.GPTNeoXModel: gpt_neox_type_to_dimension_mapping,
    hf_models.gpt_neox.modeling_gpt_neox.GPTNeoXForCausalLM: gpt_neox_lm_type_to_dimension_mapping,
    hf_models.blip.modeling_blip.BlipForQuestionAnswering: blip_type_to_dimension_mapping,
    BlipWrapper: blip_wrapper_type_to_dimension_mapping,
    MLPModel: mlp_type_to_dimension_mapping,
    MLPForClassification: mlp_classifier_type_to_dimension_mapping,
    GRUModel: gru_type_to_dimension_mapping,
    GRULMHeadModel: gru_lm_type_to_dimension_mapping,
    GRUForClassification: gru_classifier_type_to_dimension_mapping,
    # new model type goes here after defining the model files
}
#########################################################################

