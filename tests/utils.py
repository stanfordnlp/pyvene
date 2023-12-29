##################
#
# common imports
#
##################
import torch
import pandas as pd
from models.basic_utils import embed_to_distrib, top_vals, format_token
from models.configuration_alignable_model import AlignableRepresentationConfig, AlignableConfig
from models.alignable_base import AlignableModel
from models.interventions import VanillaIntervention, RotatedSpaceIntervention, LowRankRotatedSpaceIntervention
from models.mlp.modelings_mlp import MLPConfig
from models.mlp.modelings_alignable_mlp import create_mlp_classifier


##################
#
# helper functions to get golden labels 
# by manually creating counterfactual labels.
#
##################

ONE_MLP_CLEAN_RUN = lambda input_dict, mlp : \
    mlp.mlp.h[0].act(input_dict["inputs_embeds"] \
        @ mlp.mlp.h[0].ff1.weight.T) \
        @ mlp.mlp.h[0].ff2.weight.T \
        @ mlp.score.weight.T

ONE_MLP_FETCH_W1_OUT = lambda input_dict, mlp : \
    input_dict["inputs_embeds"] \
        @ mlp.mlp.h[0].ff1.weight.T

ONE_MLP_FETCH_W1_ACT = lambda input_dict, mlp : \
    mlp.mlp.h[0].act(input_dict["inputs_embeds"] \
        @ mlp.mlp.h[0].ff1.weight.T)

ONE_MLP_FETCH_W2_OUT = lambda input_dict, mlp : \
    mlp.mlp.h[0].act(input_dict["inputs_embeds"] \
        @ mlp.mlp.h[0].ff1.weight.T) \
        @ mlp.mlp.h[0].ff2.weight.T

ONE_MLP_WITH_W1_OUT_RUN = lambda w1_out, mlp : \
    mlp.mlp.h[0].act(w1_act) @ mlp.mlp.h[0].ff2.weight.T @ mlp.score.weight.T

ONE_MLP_WITH_W1_ACT_RUN = lambda w1_act, mlp : \
    w1_act @ mlp.mlp.h[0].ff2.weight.T @ mlp.score.weight.T

ONE_MLP_WITH_W2_OUT_RUN = lambda w2_out, mlp : \
    w2_out @ mlp.score.weight.T