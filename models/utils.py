import torch, random
from torch import nn
import numpy as np

from models.constants import CONST_TRANSFORMER_TOPOLOGICAL_ORDER
from models.llama.modelings_alignable_llama import *
from models.gpt2.modelings_alignable_gpt2 import *
from models.gpt_neo.modelings_alignable_gpt_neo import *
from models.gpt_neox.modelings_alignable_gpt_neox import *


lsm = nn.LogSoftmax(dim=2)
sm = nn.Softmax(dim=2)


#########################################################################
"""
Below are functions that you need to modify if you add
a new model arch type in this library.

We put them in front so it is easier to keep track of
things that need to be changed.
"""

import transformers.models as hf_models

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
    # new model type goes here after defining the model files
}


output_to_subcomponent_fn_mapping = {
    hf_models.gpt2.modeling_gpt2.GPT2Model: gpt2_output_to_subcomponent,
    hf_models.gpt2.modeling_gpt2.GPT2LMHeadModel: gpt2_output_to_subcomponent,
    hf_models.llama.modeling_llama.LlamaModel: llama_output_to_subcomponent,
    hf_models.llama.modeling_llama.LlamaForCausalLM: llama_output_to_subcomponent,
    hf_models.gpt_neo.modeling_gpt_neo.GPTNeoModel: gpt_neo_output_to_subcomponent,
    hf_models.gpt_neo.modeling_gpt_neo.GPTNeoForCausalLM: gpt_neo_output_to_subcomponent,
    hf_models.gpt_neox.modeling_gpt_neox.GPTNeoXModel: gpt_neox_output_to_subcomponent,
    hf_models.gpt_neox.modeling_gpt_neox.GPTNeoXForCausalLM: gpt_neox_output_to_subcomponent,
    # new model type goes here after defining the model files
}


scatter_intervention_output_fn_mapping = {
    hf_models.gpt2.modeling_gpt2.GPT2Model: gpt2_scatter_intervention_output,
    hf_models.gpt2.modeling_gpt2.GPT2LMHeadModel: gpt2_scatter_intervention_output,
    hf_models.llama.modeling_llama.LlamaModel: llama_scatter_intervention_output,
    hf_models.llama.modeling_llama.LlamaForCausalLM: llama_scatter_intervention_output,
    hf_models.gpt_neo.modeling_gpt_neo.GPTNeoModel: gpt_neo_scatter_intervention_output,
    hf_models.gpt_neo.modeling_gpt_neo.GPTNeoForCausalLM: gpt_neo_scatter_intervention_output,
    hf_models.gpt_neox.modeling_gpt_neox.GPTNeoXModel: gpt_neox_scatter_intervention_output,
    hf_models.gpt_neox.modeling_gpt_neox.GPTNeoXForCausalLM: gpt_neox_scatter_intervention_output,
    # new model type goes here after defining the model files
}


def get_internal_model_type(model):
    """Return the model type"""
    return type(model)


def is_transformer(model):
    """Determine if this is a transformer model"""
    return True

        
def embed_to_distrib(model, embed, log=False, logits=False):
    """Convert an embedding to a distribution over the vocabulary"""
    if "gpt2" in model.config.architectures[0].lower():
        with torch.inference_mode():
            vocab = torch.matmul(embed, model.wte.weight.t())
            if logits:
                return vocab
            return lsm(vocab) if log else sm(vocab)
    elif "llama" in model.config.architectures[0].lower():
        assert False, "Support for LLaMA is not here yet"
            
            
"""
Above are functions that you need to modify if you add
a new model arch type in this library.

We put them in front so it is easier to keep track of
things that need to be changed.
"""
#########################################################################


def print_forward_hooks(main_module):
    """Function to print forward hooks of a module and its sub-modules"""
    for name, submodule in main_module.named_modules():
        if hasattr(submodule, "_forward_hooks") and submodule._forward_hooks:
            print(f"Module: {name if name else 'Main Module'}")
            for hook_id, hook in submodule._forward_hooks.items():
                print(f"  ID: {hook_id}, Hook: {hook}")

        if hasattr(submodule, "_forward_pre_hooks") and submodule._forward_hooks:
            print(f"Module: {name if name else 'Main Module'}")
            for hook_id, hook in submodule._forward_pre_hooks.items():
                print(f"  ID: {hook_id}, Hook: {hook}")
                

def remove_forward_hooks(main_module: nn.Module):
    """Function to remove all forward and pre-forward hooks from a module and its sub-modules."""
    
    # Remove forward hooks
    for _, submodule in main_module.named_modules():
        if hasattr(submodule, "_forward_hooks"):
            hooks = list(submodule._forward_hooks.keys())  # Get a list of hook IDs
            for hook_id in hooks:
                submodule._forward_hooks.pop(hook_id)
        
        # Remove pre-forward hooks
        if hasattr(submodule, "_forward_pre_hooks"):
            pre_hooks = list(submodule._forward_pre_hooks.keys())  # Get a list of pre-hook IDs
            for pre_hook_id in pre_hooks:
                submodule._forward_pre_hooks.pop(pre_hook_id)
                
def set_seed(seed: int):
    """Set seed. Deprecate soon since it is in the huggingface library"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid_boundary(_input, boundary_x, boundary_y, temperature):
    """Generate sigmoid mask"""
    return torch.sigmoid((_input - boundary_x) / temperature) * \
        torch.sigmoid((boundary_y - _input) / temperature)


def harmonic_sigmoid_boundary(_input, boundary_x, boundary_y, temperature):
    """Generate harmonic sigmoid mask"""
    return (_input<=boundary_x)*torch.sigmoid((_input - boundary_x) / temperature) + \
    (_input>=boundary_y)*torch.sigmoid((boundary_y - _input) / temperature) + \
    ((_input>boundary_x)&(_input<boundary_y))*torch.sigmoid(
        (0.5 * (torch.abs(_input - boundary_x)**(-1) + torch.abs(_input - boundary_y)**(-1)))**(-1) / temperature
    )


def count_parameters(model):
    """Count parameters of a model that require gradients"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def random_permutation_matrix(n):
    """Generate a random permutation matrix"""
    P = torch.eye(n)
    perm = torch.randperm(n)
    P = P[perm]
    
    return P


def closeness_to_permutation_loss(R):
    """Measure how close a rotation m is close to a permutation m"""
    row_sum_diff = torch.abs(R.sum(dim=1) - 1.0).mean()
    col_sum_diff = torch.abs(R.sum(dim=0) - 1.0).mean()
    entry_diff = (R * (1 - R)).mean()
    loss = .5 * (row_sum_diff + col_sum_diff) + entry_diff
    return loss


def format_token(tokenizer, tok):
    """Format the token for some path patching experiment to show decoding diff"""
    return tokenizer.decode(tok).replace(" ", "_").replace("\n", "\\n")


def top_vals(tokenizer, res, n=10):
    """Pretty print the top n values of a distribution over the vocabulary"""
    top_values, top_indices = torch.topk(res, n)
    for i in range(len(top_values)):
        tok = format_token(tokenizer, top_indices[i].item())
        print(f"{tok:<20} {top_values[i].item()}")
        

def getattr_for_torch_module(
    model,
    parameter_name
):
    """Recursively fetch the model based on the name"""
    current_module = model
    for param in parameter_name.split("."):
        if "[" in param:
            current_module = getattr(
                current_module, param.split("[")[0]
            )[int(param.split("[")[-1].strip("]"))]
        else:
            current_module = getattr(current_module, param)
    return current_module
    

def get_alignable_dimension(
    model, representation
) -> int:
    """Based on the representation, get the aligning dimension size"""
    dimension_proposals = type_to_dimension_mapping[
        get_internal_model_type(model)
    ][
        representation.alignable_representation_type
    ]
    for proposal in dimension_proposals:
        if "*" in proposal:
            # often constant multiplier with MLP
            dimension = getattr_for_torch_module(
                model,
                proposal.split("*")[0]
            ) * int(proposal.split("*")[1])
        elif "/" in proposal:
            # often split by head number
            dimension = int(getattr_for_torch_module(
                model,
                proposal.split("/")[0]
            ) / getattr_for_torch_module(
                model,
                proposal.split("/")[1]
            ))
        else:
            dimension = getattr_for_torch_module(
                model,
                proposal
            )
        if dimension is not None:
            return dimension * int(representation.max_number_of_units)

    assert False


def get_alignable_module_hook(
    model, representation
) -> nn.Module:
    """Render the intervening module with a hook"""
    type_info = type_to_module_mapping[
        get_internal_model_type(model)
    ][
        representation.alignable_representation_type
    ]
    parameter_name = type_info[0]
    hook_type = type_info[1]
    if "%s" in parameter_name:
        # we assume it is for the layer.
        parameter_name = parameter_name % (representation.alignable_layer)
        
    module = getattr_for_torch_module(
        model,
        parameter_name
    )
    module_hook = getattr(module, hook_type)
    
    return module_hook


def sort_alignables_by_topological_order(
    model,
    alignable_representations
):
    """Sort the intervention with topology in transformer arch"""
    if is_transformer(model):
        scores = {}
        for k, _ in alignable_representations.items():
            l = int(k.split('.')[1]) + 1
            r = CONST_TRANSFORMER_TOPOLOGICAL_ORDER.index(k.split('.')[3])
            scores[k] = l*r
        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x])
        return sorted_keys
    assert False
    

class HandlerList():
    """General class to set hooks and set off hooks"""
    def __init__(self, handlers):
        self.handlers = handlers

    def remove(self):
        for handler in self.handlers:
            handler.remove()
    
    def extend(
        self, 
        new_handlers
    ):
        self.handlers.extend(new_handlers.handlers)
        return self


def gather_neurons(
    tensor_input,
    alignable_unit,
    unit_locations
):
    """Gather intervening neurons"""
    if alignable_unit in {"pos", "h"}:
        assert tensor_input.shape[0] == \
                unit_locations.shape[0]
        
        tensor_output = torch.gather(
            tensor_input, 1, 
            unit_locations.reshape(
                *unit_locations.shape, 
                *(1,)*(len(tensor_input.shape)-2)
            ).expand(
                -1, -1, *tensor_input.shape[2:]
            )
        )
        
        return tensor_output
    else:
        assert False, f"Not Implemented Gathering with Unit = {alignable_unit}"

        
def bsd_to_b_sd(tensor):
    """
    Convert a tensor of shape (b, s, d) to (b, s*d).
    """
    b, s, d = tensor.shape
    return tensor.reshape(b, s*d)


def b_sd_to_bsd(tensor, d):
    """
    Convert a tensor of shape (b, s*d) back to (b, s, d).
    """
    b, sd = tensor.shape
    s = sd // d
    return tensor.reshape(b, s, d)


def bhsd_to_bs_hd(tensor):
    """
    Convert a tensor of shape (b, h, s, d) to (b, s, h*d).
    """
    b, h, s, d = tensor.shape
    return tensor.permute(0, 2, 1, 3).reshape(b, s, h*d)


def bs_hd_to_bhsd(tensor, d):
    """
    Convert a tensor of shape (b, s, h*d) back to (b, h, s, d).
    """
    b, s, hd = tensor.shape
    h = hd // d
    return tensor.reshape(b, s, h, d).permute(0, 2, 1, 3)

       
def do_intervention(
    base_representation,
    source_representation,
    intervention
):
    """Do the actual intervention"""
    d = base_representation.shape[-1]

    # flatten
    if len(base_representation.shape) == 3:
        # b, num_unit (pos), d -> b, num_unit*d
        base_representation_f = bsd_to_b_sd(base_representation)
        source_representation_f = bsd_to_b_sd(source_representation)
    elif len(base_representation.shape) == 4:
        # b, num_unit (h), s, d -> b, s, num_unit*d
        base_representation_f = bhsd_to_bs_hd(base_representation)
        source_representation_f = bhsd_to_bs_hd(source_representation)

    intervened_representation = intervention(
        base_representation_f, source_representation_f
    )

    # unflatten
    if len(base_representation.shape) == 3:
        intervened_representation = b_sd_to_bsd(intervened_representation, d)
    elif len(base_representation.shape) == 4:
        intervened_representation = bs_hd_to_bhsd(intervened_representation, d)

    return intervened_representation


def simple_output_to_subcomponent(
    output, alignable_representation_type, model_config
):
    """This is an oversimplied version for demo"""
    return output


def simple_scatter_intervention_output(
    original_output, intervened_representation,
    alignable_representation_type,
    unit_locations, model_config
):
    """This is an oversimplied version for demo"""
    for batch_i, locations in enumerate(unit_locations):
        original_output[
            batch_i, locations
        ] = intervened_representation[batch_i]

