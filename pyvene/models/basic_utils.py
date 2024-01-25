"""
Basic Utils
"""
import os
import copy
import random
import importlib
import torch

from torch import nn
import numpy as np


lsm = nn.LogSoftmax(dim=2)
sm = nn.Softmax(dim=2)


def get_type_from_string(type_str):
    """Help function to convert string to type"""
    # Remove <class ' and '> from the string
    type_str = type_str.replace("<class '", "").replace("'>", "")

    # Split the string into module and class name
    module_name, class_name = type_str.rsplit(".", 1)

    # Import the module
    module = importlib.import_module(module_name)

    # Get the class
    cls = getattr(module, class_name)

    return cls


def create_directory(path):
    """Create directory if not exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    else:
        print(f"Directory '{path}' already exists.")


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


def set_seed(seed: int):
    """Set seed. Deprecate soon since it is in the huggingface library"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid_boundary(_input, boundary_x, boundary_y, temperature):
    """Generate sigmoid mask"""
    return torch.sigmoid((_input - boundary_x) / temperature) * torch.sigmoid(
        (boundary_y - _input) / temperature
    )


def harmonic_sigmoid_boundary(_input, boundary_x, boundary_y, temperature):
    """Generate harmonic sigmoid mask"""
    return (
        (_input <= boundary_x) * torch.sigmoid((_input - boundary_x) / temperature)
        + (_input >= boundary_y) * torch.sigmoid((boundary_y - _input) / temperature)
        + ((_input > boundary_x) & (_input < boundary_y))
        * torch.sigmoid(
            (
                0.5
                * (
                    torch.abs(_input - boundary_x) ** (-1)
                    + torch.abs(_input - boundary_y) ** (-1)
                )
            )
            ** (-1)
            / temperature
        )
    )


def count_parameters(model):
    """Count parameters of a model that require gradients"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def random_permutation_matrix(n):
    """Generate a random permutation matrix"""
    _p = torch.eye(n)
    perm = torch.randperm(n)
    _p = _p[perm]

    return _p


def closeness_to_permutation_loss(rotation):
    """Measure how close a rotation m is close to a permutation m"""
    row_sum_diff = torch.abs(rotation.sum(dim=1) - 1.0).mean()
    col_sum_diff = torch.abs(rotation.sum(dim=0) - 1.0).mean()
    entry_diff = (rotation * (1 - rotation)).mean()
    loss = 0.5 * (row_sum_diff + col_sum_diff) + entry_diff
    return loss


def format_token(tokenizer, tok):
    """Format the token for some path patching experiment to show decoding diff"""
    return tokenizer.decode(tok).replace(" ", "_").replace("\n", "\\n")


def top_vals(tokenizer, res, n=10, return_results=False):
    """Pretty print the top n values of a distribution over the vocabulary"""
    top_values, top_indices = torch.topk(res, n)
    ret = []
    for i, _ in enumerate(top_values):
        tok = format_token(tokenizer, top_indices[i].item())
        ret += [(tok, top_values[i].item())]
        if not return_results:
            print(f"{tok:<20} {top_values[i].item()}")
    if return_results:
        return ret
        
def get_list_depth(lst):
    """Return the max depth of the input list"""
    if isinstance(lst, list):
        return 1 + max((get_list_depth(item) for item in lst), default=0)
    return 0


def get_batch_size(model_input):
    """
    Get batch size based on the input
    """
    if isinstance(model_input, torch.Tensor):
        batch_size = model_input.shape[0]
    else:
        for _, v in model_input.items():
            batch_size = v.shape[0]
            break
    return batch_size


def GET_LOC(
    LOC,
    unit="h.pos",
    batch_size=1,
):
    """
    From simple locale to nested one.
    """
    if unit == "h.pos":
        return [ 
                   [ 
                       [ 
                           [LOC[0]] 
                       ] * batch_size, 
                       [ 
                           [LOC[1]] 
                       ] * batch_size 
                   ] 
               ]
    else:
        raise NotImplementedError(
            f"{unit} is not supported."
        )