##################
#
# common imports
#
##################
import os, shutil, torch, random, uuid
import pandas as pd
import numpy as np
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
from pyvene.models.interventions import *
from pyvene.models.mlp.modelings_mlp import MLPConfig
from pyvene.models.mlp.modelings_intervenable_mlp import create_mlp_classifier
from pyvene.models.gpt2.modelings_intervenable_gpt2 import create_gpt2_lm


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


"""
forward calls to fetch activations or run with cached activations
"""


def DO_GPT2_INTERVENTION(name, orig_hidden_states, INTERVENTION_ACTIVATIONS):
    if name in INTERVENTION_ACTIVATIONS:
        return INTERVENTION_ACTIVATIONS[name]
    return orig_hidden_states


def GPT2_SELF_ATTENTION_RUN(
    self_attn, hidden_states, i, CACHE_ACTIVATIONS, INTERVENTION_ACTIVATIONS
):
    query, key, value = self_attn.c_attn(hidden_states).split(
        self_attn.split_size, dim=2
    )

    query = DO_GPT2_INTERVENTION(f"{i}.query_output", query, INTERVENTION_ACTIVATIONS)
    CACHE_ACTIVATIONS[f"{i}.query_output"] = query
    key = DO_GPT2_INTERVENTION(f"{i}.key_output", key, INTERVENTION_ACTIVATIONS)
    CACHE_ACTIVATIONS[f"{i}.key_output"] = key
    value = DO_GPT2_INTERVENTION(f"{i}.value_output", value, INTERVENTION_ACTIVATIONS)
    CACHE_ACTIVATIONS[f"{i}.value_output"] = value

    head_query = self_attn._split_heads(query, self_attn.num_heads, self_attn.head_dim)
    head_key = self_attn._split_heads(key, self_attn.num_heads, self_attn.head_dim)
    head_value = self_attn._split_heads(value, self_attn.num_heads, self_attn.head_dim)

    head_query = DO_GPT2_INTERVENTION(
        f"{i}.head_query_output", head_query, INTERVENTION_ACTIVATIONS
    )
    CACHE_ACTIVATIONS[f"{i}.head_query_output"] = head_query
    head_key = DO_GPT2_INTERVENTION(
        f"{i}.head_key_output", head_key, INTERVENTION_ACTIVATIONS
    )
    CACHE_ACTIVATIONS[f"{i}.head_key_output"] = head_key
    head_value = DO_GPT2_INTERVENTION(
        f"{i}.head_value_output", head_value, INTERVENTION_ACTIVATIONS
    )
    CACHE_ACTIVATIONS[f"{i}.head_value_output"] = head_value

    head_attention_value_output, attn_weights = self_attn._attn(
        head_query, head_key, head_value
    )

    head_attention_value_output = DO_GPT2_INTERVENTION(
        f"{i}.head_attention_value_output",
        head_attention_value_output,
        INTERVENTION_ACTIVATIONS,
    )
    CACHE_ACTIVATIONS[f"{i}.head_attention_value_output"] = head_attention_value_output

    attn_value_output = self_attn._merge_heads(
        head_attention_value_output, self_attn.num_heads, self_attn.head_dim
    )
    attn_value_output = DO_GPT2_INTERVENTION(
        f"{i}.attention_value_output", attn_value_output, INTERVENTION_ACTIVATIONS
    )
    CACHE_ACTIVATIONS[f"{i}.attention_value_output"] = attn_value_output
    attn_output = self_attn.c_proj(attn_value_output)
    attn_output = self_attn.resid_dropout(attn_output)

    return attn_output


def GPT2_MLP_RUN(mlp, hidden_states, i, CACHE_ACTIVATIONS, INTERVENTION_ACTIVATIONS):
    hidden_states_c_fc = mlp.c_fc(hidden_states)
    hidden_states_act = mlp.act(hidden_states_c_fc)

    hidden_states_act = DO_GPT2_INTERVENTION(
        f"{i}.mlp_activation", hidden_states_act, INTERVENTION_ACTIVATIONS
    )
    CACHE_ACTIVATIONS[f"{i}.mlp_activation"] = hidden_states_act

    hidden_states_c_proj = mlp.c_proj(hidden_states_act)
    hidden_states_c_proj = mlp.dropout(hidden_states_c_proj)
    return hidden_states_c_proj


def GPT2_BLOCK_RUN(
    block, hidden_states, i, CACHE_ACTIVATIONS, INTERVENTION_ACTIVATIONS
):
    # self attention + residual
    residual = hidden_states
    hidden_states = block.ln_1(hidden_states)

    hidden_states = DO_GPT2_INTERVENTION(
        f"{i}.attention_input", hidden_states, INTERVENTION_ACTIVATIONS
    )
    CACHE_ACTIVATIONS[f"{i}.attention_input"] = hidden_states

    attn_outputs = GPT2_SELF_ATTENTION_RUN(
        block.attn, hidden_states, i, CACHE_ACTIVATIONS, INTERVENTION_ACTIVATIONS
    )

    attn_outputs = DO_GPT2_INTERVENTION(
        f"{i}.attention_output", attn_outputs, INTERVENTION_ACTIVATIONS
    )
    CACHE_ACTIVATIONS[f"{i}.attention_output"] = attn_outputs

    attn_output = attn_outputs
    # residual connection
    hidden_states = attn_output + residual

    # mlp + residual
    residual = hidden_states
    hidden_states = block.ln_2(hidden_states)

    hidden_states = DO_GPT2_INTERVENTION(
        f"{i}.mlp_input", hidden_states, INTERVENTION_ACTIVATIONS
    )
    CACHE_ACTIVATIONS[f"{i}.mlp_input"] = hidden_states

    feed_forward_hidden_states = GPT2_MLP_RUN(
        block.mlp, hidden_states, i, CACHE_ACTIVATIONS, INTERVENTION_ACTIVATIONS
    )

    feed_forward_hidden_states = DO_GPT2_INTERVENTION(
        f"{i}.mlp_output", feed_forward_hidden_states, INTERVENTION_ACTIVATIONS
    )
    CACHE_ACTIVATIONS[f"{i}.mlp_output"] = feed_forward_hidden_states

    # residual connection
    hidden_states = residual + feed_forward_hidden_states

    return hidden_states


def GPT2_RUN(gpt2, input_ids, CACHE_ACTIVATIONS, INTERVENTION_ACTIVATIONS):
    """
    We basically explicitly do the gpt2 forward here.
    """
    device = gpt2.device
    input_shape = input_ids.shape

    # embed + pos_embed
    inputs_embeds = gpt2.transformer.wte(input_ids)
    position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0)
    position_embeds = gpt2.transformer.wpe(position_ids)
    hidden_states = inputs_embeds + position_embeds

    for i, block in enumerate(gpt2.transformer.h):
        hidden_states = DO_GPT2_INTERVENTION(
            f"{i}.block_input", hidden_states, INTERVENTION_ACTIVATIONS
        )
        CACHE_ACTIVATIONS[f"{i}.block_input"] = hidden_states

        hidden_states = GPT2_BLOCK_RUN(
            block, hidden_states, i, CACHE_ACTIVATIONS, INTERVENTION_ACTIVATIONS
        )

        hidden_states = DO_GPT2_INTERVENTION(
            f"{i}.block_output", hidden_states, INTERVENTION_ACTIVATIONS
        )
        CACHE_ACTIVATIONS[f"{i}.block_output"] = hidden_states

    hidden_states = gpt2.transformer.ln_f(hidden_states)
    lm_logits = gpt2.lm_head(hidden_states)
    return lm_logits
