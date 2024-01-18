from .constants import *
from .llama.modelings_intervenable_llama import *
from .mistral.modelings_intervenable_mistral import *
from .gpt2.modelings_intervenable_gpt2 import *
from .gpt_neo.modelings_intervenable_gpt_neo import *
from .gpt_neox.modelings_intervenable_gpt_neox import *
from .mlp.modelings_intervenable_mlp import *
from .gru.modelings_intervenable_gru import *
from .blip.modelings_intervenable_blip import *
from .bert.modelings_intervenable_bert import *
from .roberta.modelings_intervenable_roberta import *
from .electra.modelings_intervenable_electra import *
from .xlm.modelings_intervenable_xlm import *
from .mistral.modelings_intervenable_mistral import *
from .t5.modelings_intervenable_t5 import *
from .mixtral.modelings_intervenable_mixtral import *
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
from .mlp.modelings_mlp import MLPModel, MLPForClassification
from .gru.modelings_gru import GRUModel, GRULMHeadModel, GRUForClassification
from .backpack_gpt2.modeling_backpack_gpt2 import BackpackGPT2LMHeadModel

global type_to_module_mapping
global type_to_dimension_mapping
global output_to_subcomponent_fn_mapping
global scatter_intervention_output_fn_mapping


type_to_module_mapping = {
    # gpt2
    hf_models.gpt2.modeling_gpt2.GPT2Model: gpt2_type_to_module_mapping,
    hf_models.gpt2.modeling_gpt2.GPT2LMHeadModel: gpt2_lm_type_to_module_mapping,
    # llama
    hf_models.llama.modeling_llama.LlamaModel: llama_type_to_module_mapping,
    hf_models.llama.modeling_llama.LlamaForCausalLM: llama_lm_type_to_module_mapping,
    # gpt-neo
    hf_models.gpt_neo.modeling_gpt_neo.GPTNeoModel: gpt_neo_type_to_module_mapping,
    hf_models.gpt_neo.modeling_gpt_neo.GPTNeoForCausalLM: gpt_neo_lm_type_to_module_mapping,
    hf_models.gpt_neox.modeling_gpt_neox.GPTNeoXModel: gpt_neox_type_to_module_mapping,
    hf_models.gpt_neox.modeling_gpt_neox.GPTNeoXForCausalLM: gpt_neox_lm_type_to_module_mapping,
    # blip
    hf_models.blip.modeling_blip.BlipForQuestionAnswering: blip_type_to_module_mapping,
    BlipWrapper: blip_wrapper_type_to_module_mapping,
    # mlp
    MLPModel: mlp_type_to_module_mapping,
    MLPForClassification: mlp_classifier_type_to_module_mapping,
    #gru
    GRUModel: gru_type_to_module_mapping,
    GRULMHeadModel: gru_lm_type_to_module_mapping,
    GRUForClassification: gru_classifier_type_to_module_mapping,
    # bert
    hf_models.bert.modeling_bert.BertLMHeadModel: bert_generic_type_to_module_mapping,
    hf_models.bert.modeling_bert.BertForMaskedLM: bert_generic_type_to_module_mapping,
    hf_models.bert.modeling_bert.BertForSequenceClassification: bert_generic_type_to_module_mapping,
    # roberta
    hf_models.roberta.modeling_roberta.RobertaForCausalLM: roberta_generic_type_to_module_mapping,
    hf_models.roberta.modeling_roberta.RobertaForMaskedLM: roberta_generic_type_to_module_mapping,
    hf_models.roberta.modeling_roberta.RobertaForSequenceClassification: roberta_generic_type_to_module_mapping,
    # electra
    hf_models.electra.modeling_electra.ElectraForCausalLM: electra_generic_type_to_module_mapping,
    hf_models.electra.modeling_electra.ElectraForMaskedLM: electra_generic_type_to_module_mapping,
    hf_models.electra.modeling_electra.ElectraForSequenceClassification: electra_generic_type_to_module_mapping,
    # xlm
    hf_models.xlm.modeling_xlm.XLMWithLMHeadModel: xlm_generic_type_to_module_mapping,
    hf_models.xlm.modeling_xlm.XLMForSequenceClassification: xlm_generic_type_to_module_mapping,
    # mistral
    hf_models.mistral.modeling_mistral.MistralModel: mistral_type_to_module_mapping,
    hf_models.mistral.modeling_mistral.MistralForCausalLM: mistral_lm_type_to_module_mapping,
    # t5
    hf_models.t5.modeling_t5.T5ForConditionalGeneration: t5_lm_type_to_module_mapping,
    # mixtral
    hf_models.mixtral.modeling_mixtral.MixtralForCausalLM: mixtral_lm_type_to_module_mapping,
    # backpack
    BackpackGPT2LMHeadModel: backpack_gpt2_lm_type_to_module_mapping,
    # new model type goes here after defining the model files
}


type_to_dimension_mapping = {
    # gpt2
    hf_models.gpt2.modeling_gpt2.GPT2Model: gpt2_type_to_dimension_mapping,
    hf_models.gpt2.modeling_gpt2.GPT2LMHeadModel: gpt2_lm_type_to_dimension_mapping,
    # llama
    hf_models.llama.modeling_llama.LlamaModel: llama_type_to_dimension_mapping,
    hf_models.llama.modeling_llama.LlamaForCausalLM: llama_lm_type_to_dimension_mapping,
    # gpt-neo
    hf_models.gpt_neo.modeling_gpt_neo.GPTNeoModel: gpt_neo_type_to_dimension_mapping,
    hf_models.gpt_neo.modeling_gpt_neo.GPTNeoForCausalLM: gpt_neo_lm_type_to_dimension_mapping,
    hf_models.gpt_neox.modeling_gpt_neox.GPTNeoXModel: gpt_neox_type_to_dimension_mapping,
    hf_models.gpt_neox.modeling_gpt_neox.GPTNeoXForCausalLM: gpt_neox_lm_type_to_dimension_mapping,
    # blip
    hf_models.blip.modeling_blip.BlipForQuestionAnswering: blip_type_to_dimension_mapping,
    BlipWrapper: blip_wrapper_type_to_dimension_mapping,
    # mlp
    MLPModel: mlp_type_to_dimension_mapping,
    MLPForClassification: mlp_classifier_type_to_dimension_mapping,
    # gru
    GRUModel: gru_type_to_dimension_mapping,
    GRULMHeadModel: gru_lm_type_to_dimension_mapping,
    GRUForClassification: gru_classifier_type_to_dimension_mapping,
    # bert
    hf_models.bert.modeling_bert.BertLMHeadModel: bert_generic_type_to_dimension_mapping,
    hf_models.bert.modeling_bert.BertForMaskedLM: bert_generic_type_to_dimension_mapping,
    hf_models.bert.modeling_bert.BertForSequenceClassification: bert_generic_type_to_dimension_mapping,
    # roberta
    hf_models.roberta.modeling_roberta.RobertaForCausalLM: roberta_generic_type_to_dimension_mapping,
    hf_models.roberta.modeling_roberta.RobertaForMaskedLM: roberta_generic_type_to_dimension_mapping,
    hf_models.roberta.modeling_roberta.RobertaForSequenceClassification: roberta_generic_type_to_dimension_mapping,
    # electra
    hf_models.electra.modeling_electra.ElectraForCausalLM: electra_generic_type_to_dimension_mapping,
    hf_models.electra.modeling_electra.ElectraForMaskedLM: electra_generic_type_to_dimension_mapping,
    hf_models.electra.modeling_electra.ElectraForSequenceClassification: electra_generic_type_to_dimension_mapping,
    # xlm
    hf_models.xlm.modeling_xlm.XLMWithLMHeadModel: xlm_generic_type_to_dimension_mapping,
    hf_models.xlm.modeling_xlm.XLMForSequenceClassification: xlm_generic_type_to_dimension_mapping,
    # mistral
    hf_models.mistral.modeling_mistral.MistralModel: mistral_type_to_dimension_mapping,
    hf_models.mistral.modeling_mistral.MistralForCausalLM: mistral_lm_type_to_dimension_mapping,
    # t5
    hf_models.t5.modeling_t5.T5ForConditionalGeneration: t5_lm_type_to_dimension_mapping,
    # mixtral
    hf_models.mixtral.modeling_mixtral.MixtralForCausalLM: mixtral_lm_type_to_dimension_mapping,
    # backpack
    BackpackGPT2LMHeadModel: backpack_gpt2_lm_type_to_dimension_mapping,
    # new model type goes here after defining the model files
}
#########################################################################
