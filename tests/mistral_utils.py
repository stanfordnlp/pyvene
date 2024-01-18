##################
#
# common imports
#
##################
import os, shutil, torch, random, uuid, math
import pandas as pd
import numpy as np
from transformers import MistralConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

import torch.nn.functional as F


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
from pyvene.models.mistral.modelings_intervenable_mistral import create_mistral


##################
#
# helper functions to get golden labels
# by manually creating counterfactual labels.
#
##################

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

"""
forward calls to fetch activations or run with cached activations
"""


def DO_MISTRAL_INTERVENTION(name, orig_hidden_states, INTERVENTION_ACTIVATIONS):
    if name in INTERVENTION_ACTIVATIONS:
        return INTERVENTION_ACTIVATIONS[name]
    return orig_hidden_states


def MISTRAL_SELF_ATTENTION_RUN(
    self_attn, hidden_states, position_ids, attention_mask, i, CACHE_ACTIVATIONS, INTERVENTION_ACTIVATIONS
):
    b, s, _ = hidden_states.size()
    query = self_attn.q_proj(hidden_states)
    key = self_attn.k_proj(hidden_states)
    value = self_attn.v_proj(hidden_states)
    
    query = DO_MISTRAL_INTERVENTION(f"{i}.query_output", query, INTERVENTION_ACTIVATIONS)
    CACHE_ACTIVATIONS[f"{i}.query_output"] = query
    key = DO_MISTRAL_INTERVENTION(f"{i}.key_output", key, INTERVENTION_ACTIVATIONS)
    CACHE_ACTIVATIONS[f"{i}.key_output"] = key
    value = DO_MISTRAL_INTERVENTION(f"{i}.value_output", value, INTERVENTION_ACTIVATIONS)
    CACHE_ACTIVATIONS[f"{i}.value_output"] = value

    head_query = query.view(b, s, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)
    head_key = key.view(b, s, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)
    head_value = value.view(b, s, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)

    head_query = DO_MISTRAL_INTERVENTION(
        f"{i}.head_query_output", head_query, INTERVENTION_ACTIVATIONS
    )
    CACHE_ACTIVATIONS[f"{i}.head_query_output"] = head_query
    head_key = DO_MISTRAL_INTERVENTION(
        f"{i}.head_key_output", head_key, INTERVENTION_ACTIVATIONS
    )
    CACHE_ACTIVATIONS[f"{i}.head_key_output"] = head_key
    head_value = DO_MISTRAL_INTERVENTION(
        f"{i}.head_value_output", head_value, INTERVENTION_ACTIVATIONS
    )
    CACHE_ACTIVATIONS[f"{i}.head_value_output"] = head_value
    
    cos, sin = self_attn.rotary_emb(head_value, seq_len=s)
    head_query, head_key = apply_rotary_pos_emb(head_query, head_key, cos, sin, position_ids)
    # Assume n_kv_heads == n_heads
    
    attn_weights = torch.matmul(head_query, head_key.transpose(2, 3)) / math.sqrt(self_attn.head_dim)
    if attention_mask.size() != (b, 1, s, s):
        raise ValueError("Attention Mask")
    attn_weights = attn_weights + attention_mask
    # Upcast attention to fp32
    attn_weights = F.softmax(attn_weights, dim=-1,dtype=torch.float32).to(head_query.dtype)
    attn_weights = F.dropout(attn_weights, p=self_attn.attention_dropout, training=self_attn.training)
    head_attention_value_output = torch.matmul(attn_weights, head_value)

    head_attention_value_output = DO_MISTRAL_INTERVENTION(
        f"{i}.head_attention_value_output",
        head_attention_value_output,
        INTERVENTION_ACTIVATIONS,
    )
    CACHE_ACTIVATIONS[f"{i}.head_attention_value_output"] = head_attention_value_output
    
    attn_value_output = head_attention_value_output.transpose(1, 2).contiguous()
    attn_value_output = attn_value_output.reshape(b, s, self_attn.hidden_size)

    attn_value_output = DO_MISTRAL_INTERVENTION(
        f"{i}.attention_value_output", attn_value_output, INTERVENTION_ACTIVATIONS
    )
    CACHE_ACTIVATIONS[f"{i}.attention_value_output"] = attn_value_output
    attn_output = self_attn.o_proj(attn_value_output)
    return attn_output


def MISTRAL_MLP_RUN(mlp, hidden_states, i, CACHE_ACTIVATIONS, INTERVENTION_ACTIVATIONS):
    hidden_states_c_fc = mlp.gate_proj(hidden_states)
    hidden_states_act = mlp.act_fn(hidden_states_c_fc)

    hidden_states_act = DO_MISTRAL_INTERVENTION(
        f"{i}.mlp_activation", hidden_states_act, INTERVENTION_ACTIVATIONS
    )
    CACHE_ACTIVATIONS[f"{i}.mlp_activation"] = hidden_states_act
    
    hidden_states_c_proj = mlp.down_proj(hidden_states_act * mlp.up_proj(hidden_states))
    return hidden_states_c_proj


def MISTRAL_BLOCK_RUN(
    block, hidden_states, position_ids, attention_mask, i, CACHE_ACTIVATIONS, INTERVENTION_ACTIVATIONS
):
    # self attention + residual
    residual = hidden_states
    hidden_states = block.input_layernorm(hidden_states)

    hidden_states = DO_MISTRAL_INTERVENTION(
        f"{i}.attention_input", hidden_states, INTERVENTION_ACTIVATIONS
    )
    CACHE_ACTIVATIONS[f"{i}.attention_input"] = hidden_states

    attn_outputs = MISTRAL_SELF_ATTENTION_RUN(
        block.self_attn, hidden_states, attention_mask, position_ids, i, CACHE_ACTIVATIONS, INTERVENTION_ACTIVATIONS
    )

    attn_outputs = DO_MISTRAL_INTERVENTION(
        f"{i}.attention_output", attn_outputs, INTERVENTION_ACTIVATIONS
    )
    CACHE_ACTIVATIONS[f"{i}.attention_output"] = attn_outputs

    attn_output = attn_outputs
    # residual connection
    hidden_states = attn_output + residual

    # mlp + residual
    residual = hidden_states
    hidden_states = block.post_attention_layernorm(hidden_states)

    hidden_states = DO_MISTRAL_INTERVENTION(
        f"{i}.mlp_input", hidden_states, INTERVENTION_ACTIVATIONS
    )
    CACHE_ACTIVATIONS[f"{i}.mlp_input"] = hidden_states

    feed_forward_hidden_states = MISTRAL_MLP_RUN(
        block.mlp, hidden_states, i, CACHE_ACTIVATIONS, INTERVENTION_ACTIVATIONS
    )

    feed_forward_hidden_states = DO_MISTRAL_INTERVENTION(
        f"{i}.mlp_output", feed_forward_hidden_states, INTERVENTION_ACTIVATIONS
    )
    CACHE_ACTIVATIONS[f"{i}.mlp_output"] = feed_forward_hidden_states

    # residual connection
    hidden_states = residual + feed_forward_hidden_states

    return hidden_states


def MISTRAL_RUN(mistral, input_ids, CACHE_ACTIVATIONS, INTERVENTION_ACTIVATIONS):
    """
    We basically explicitly do the Mistral forward here.
    """
    device = mistral.device
    input_shape = input_ids.shape

    # embed + pos_embed
    inputs_embeds = mistral.model.embed_tokens(input_ids)
    b, s, _ = inputs_embeds.shape
    attention_mask = _prepare_4d_causal_attention_mask(
        None,
        (b, s),
        inputs_embeds,
        0,
        sliding_window=4096,
    )
    position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0)
    hidden_states = inputs_embeds

    for i, block in enumerate(mistral.model.layers):
        hidden_states = DO_MISTRAL_INTERVENTION(
            f"{i}.block_input", hidden_states, INTERVENTION_ACTIVATIONS
        )
        CACHE_ACTIVATIONS[f"{i}.block_input"] = hidden_states

        hidden_states = MISTRAL_BLOCK_RUN(
            block, hidden_states, attention_mask, position_ids, i, CACHE_ACTIVATIONS, INTERVENTION_ACTIVATIONS
        )

        hidden_states = DO_MISTRAL_INTERVENTION(
            f"{i}.block_output", hidden_states, INTERVENTION_ACTIVATIONS
        )
        CACHE_ACTIVATIONS[f"{i}.block_output"] = hidden_states
    hidden_states = mistral.model.norm(hidden_states)
    lm_logits = mistral.lm_head(hidden_states)
    return lm_logits
