from .constants import *
from .llama.modelings_intervenable_llama import *
from .gpt2.modelings_intervenable_gpt2 import *
from .gpt_neo.modelings_intervenable_gpt_neo import *
from .gpt_neox.modelings_intervenable_gpt_neox import *
from .mlp.modelings_intervenable_mlp import *
from .gru.modelings_intervenable_gru import *
from .blip.modelings_intervenable_blip import *
from .blip.modelings_intervenable_blip_itm import *
from .backpack_gpt2.modelings_intervenable_backpack_gpt2 import *


#########################################################################
"""
Below are functions that you need to modify if you add
a new model arch type in this library.

We put them in front so it is easier to keep track of
things that need to be changed.
"""

import transformers.models as hf_models
from .blip.modelings_blip import BlipWrapper
from .blip.modelings_blip_itm import BlipITMWrapper
from .mlp.modelings_mlp import MLPModel, MLPForClassification
from .gru.modelings_gru import GRUModel, GRULMHeadModel, GRUForClassification
from .backpack_gpt2.modelings_backpack_gpt2 import BackpackGPT2LMHeadModel

global type_to_module_mapping
global type_to_dimension_mapping
global output_to_subcomponent_fn_mapping
global scatter_intervention_output_fn_mapping


type_to_module_mapping = {
    hf_models.gpt2.modeling_gpt2.GPT2Model: gpt2_type_to_module_mapping,
    hf_models.gpt2.modeling_gpt2.GPT2LMHeadModel: gpt2_lm_type_to_module_mapping,
    hf_models.gpt2.modeling_gpt2.GPT2ForSequenceClassification: gpt2_classifier_type_to_module_mapping,
    hf_models.llama.modeling_llama.LlamaModel: llama_type_to_module_mapping,
    hf_models.llama.modeling_llama.LlamaForCausalLM: llama_lm_type_to_module_mapping,
    hf_models.gpt_neo.modeling_gpt_neo.GPTNeoModel: gpt_neo_type_to_module_mapping,
    hf_models.gpt_neo.modeling_gpt_neo.GPTNeoForCausalLM: gpt_neo_lm_type_to_module_mapping,
    hf_models.gpt_neox.modeling_gpt_neox.GPTNeoXModel: gpt_neox_type_to_module_mapping,
    hf_models.gpt_neox.modeling_gpt_neox.GPTNeoXForCausalLM: gpt_neox_lm_type_to_module_mapping,
    hf_models.blip.modeling_blip.BlipForQuestionAnswering: blip_type_to_module_mapping,
    hf_models.blip.modeling_blip.BlipForImageTextRetrieval: blip_itm_type_to_module_mapping,
    BlipWrapper: blip_wrapper_type_to_module_mapping,
    BlipITMWrapper: blip_itm_wrapper_type_to_module_mapping,
    MLPModel: mlp_type_to_module_mapping,
    MLPForClassification: mlp_classifier_type_to_module_mapping,
    GRUModel: gru_type_to_module_mapping,
    GRULMHeadModel: gru_lm_type_to_module_mapping,
    GRUForClassification: gru_classifier_type_to_module_mapping,
    BackpackGPT2LMHeadModel: backpack_gpt2_lm_type_to_module_mapping,
    # new model type goes here after defining the model files
}


type_to_dimension_mapping = {
    hf_models.gpt2.modeling_gpt2.GPT2Model: gpt2_type_to_dimension_mapping,
    hf_models.gpt2.modeling_gpt2.GPT2LMHeadModel: gpt2_lm_type_to_dimension_mapping,
    hf_models.gpt2.modeling_gpt2.GPT2ForSequenceClassification: gpt2_classifier_type_to_dimension_mapping,
    hf_models.llama.modeling_llama.LlamaModel: llama_type_to_dimension_mapping,
    hf_models.llama.modeling_llama.LlamaForCausalLM: llama_lm_type_to_dimension_mapping,
    hf_models.gpt_neo.modeling_gpt_neo.GPTNeoModel: gpt_neo_type_to_dimension_mapping,
    hf_models.gpt_neo.modeling_gpt_neo.GPTNeoForCausalLM: gpt_neo_lm_type_to_dimension_mapping,
    hf_models.gpt_neox.modeling_gpt_neox.GPTNeoXModel: gpt_neox_type_to_dimension_mapping,
    hf_models.gpt_neox.modeling_gpt_neox.GPTNeoXForCausalLM: gpt_neox_lm_type_to_dimension_mapping,
    hf_models.blip.modeling_blip.BlipForQuestionAnswering: blip_type_to_dimension_mapping,
    hf_models.blip.modeling_blip.BlipForImageTextRetrieval: blip_itm_type_to_dimension_mapping,
    BlipWrapper: blip_wrapper_type_to_dimension_mapping,
    BlipITMWrapper: blip_itm_wrapper_type_to_dimension_mapping,
    MLPModel: mlp_type_to_dimension_mapping,
    MLPForClassification: mlp_classifier_type_to_dimension_mapping,
    GRUModel: gru_type_to_dimension_mapping,
    GRULMHeadModel: gru_lm_type_to_dimension_mapping,
    GRUForClassification: gru_classifier_type_to_dimension_mapping,
    BackpackGPT2LMHeadModel: backpack_gpt2_lm_type_to_dimension_mapping,
    # new model type goes here after defining the model files
}
#########################################################################
