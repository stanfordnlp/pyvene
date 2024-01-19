##################
#
# common imports
#
##################
import os, shutil, torch, random, uuid
import pandas as pd
import numpy as np
from transformers import MistralConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers import GPT2Config

import subprocess

def is_package_installed(package_name):
    try:
        # Execute 'pip list' command and capture the output
        result = subprocess.run(['pip', 'list'], stdout=subprocess.PIPE, text=True)

        # Check if package_name is in the result
        return package_name in result.stdout
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

# Replace 'pyvene' with the name of the package you want to check
package_name = 'pyvene'
if is_package_installed(package_name):
    raise RuntimeError(
        f"Remove your pip installed {package_name} before running tests.")
else:
    print(f"'{package_name}' is not installed.")
    print("PASS: pyvene is not installed. Testing local dev code.")

from pyvene.models.basic_utils import embed_to_distrib, top_vals, format_token
from pyvene.models.configuration_intervenable_model import (
    IntervenableRepresentationConfig,
    IntervenableConfig,
)
from pyvene.models.intervenable_base import IntervenableModel
from pyvene.models.interventions import (
    VanillaIntervention,
    RotatedSpaceIntervention,
    LowRankRotatedSpaceIntervention,
)
from pyvene.models.mlp.modelings_mlp import MLPConfig
from pyvene.models.mlp.modelings_intervenable_mlp import create_mlp_classifier
from pyvene.models.gpt2.modelings_intervenable_gpt2 import create_gpt2_lm
from pyvene.models.mistral.modelings_intervenable_mistral import create_mistral


##################
#
# helper functions to get golden labels
# by manually creating counterfactual labels.
#
##################

ONE_MLP_CLEAN_RUN = (
    lambda input_dict, mlp: mlp.mlp.h[0].act(
        input_dict["inputs_embeds"] @ mlp.mlp.h[0].ff1.weight.T
    )
    @ mlp.score.weight.T
)

ONE_MLP_FETCH_W1_OUT = (
    lambda input_dict, mlp: input_dict["inputs_embeds"] @ mlp.mlp.h[0].ff1.weight.T
)

ONE_MLP_FETCH_W1_ACT = lambda input_dict, mlp: mlp.mlp.h[0].act(
    input_dict["inputs_embeds"] @ mlp.mlp.h[0].ff1.weight.T
)

ONE_MLP_WITH_W1_OUT_RUN = (
    lambda w1_out, mlp: mlp.mlp.h[0].act(w1_out) @ mlp.score.weight.T
)

ONE_MLP_WITH_W1_ACT_RUN = lambda w1_act, mlp: w1_act @ mlp.score.weight.T
