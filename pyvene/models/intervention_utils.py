import json
import torch
import pprint

class InterventionState(object):
    def __init__(self, key, **kwargs):
        self.key = key
        self._timestep = [0, 0]

    @property
    def getter_timestep(self):
        return self._timestep[0]
    
    @getter_timestep.setter
    def getter_timestep(self, value):
        self._timestep[0] = value
    
    @property
    def setter_timestep(self):
        return self._timestep[1]
    
    @setter_timestep.setter
    def setter_timestep(self, value):
        self._timestep[1] = value

    def reset(self):
        self._timestep = [0, 0]

    def __repr__(self):
        return pprint.pformat(self.__dict__, indent=4)

def broadcast_tensor_v1(x, target_shape):
    # Ensure the last dimension of target_shape matches x's size
    if target_shape[-1] != x.shape[-1]:
        raise ValueError("The last dimension of target_shape must match the size of x")

    # Create a shape for reshaping x that is compatible with target_shape
    reshape_shape = [1] * (len(target_shape) - 1) + [x.shape[-1]]

    # Reshape x and then broadcast it
    x_reshaped = x.view(*reshape_shape)
    broadcasted_x = x_reshaped.expand(*target_shape)
    return broadcasted_x
    
def broadcast_tensor_v2(x, target_shape):
    # Ensure that target_shape has at least one dimension
    if len(target_shape) < 1:
        raise ValueError("Target shape must have at least one dimension")

    # Extract the first n-1 dimensions from the target shape
    target_dims_except_last = target_shape[:-1]

    # Broadcast the input tensor x to match the target_dims_except_last and keep its last dimension
    broadcasted_x = x.expand(*target_dims_except_last, x.shape[-1])

    return broadcasted_x

def _can_cast_tensor(
    subspaces
):
    tensorfiable = True
    try:
        torch.tensor(subspaces)
    except:
        tensorfiable = False
        
    return tensorfiable

def _can_use_fast(
    subspaces
):
    tensorfiable = True
    row_same_val = False
    try:
        subspaces = torch.tensor(subspaces)
        row_same_val = torch.all(subspaces == subspaces[0], axis=1).all()
    except:
        tensorfiable = False
        
    return row_same_val and tensorfiable

def _do_intervention_by_swap(
    base,
    source,
    mode="interchange",
    interchange_dim=None,
    subspaces=None,
    subspace_partition=None,
    use_fast=False,
):
    """The basic do function that guards interventions"""
    if mode == "collect":
        assert source is None
    else:
        # auto broadcast
        if base.shape != source.shape:
            try:
                source = broadcast_tensor_v1(source, base.shape)
            except:
                raise ValueError(
                    f"source with shape {source.shape} cannot be broadcasted "
                    f"into base with shape {base.shape}."
                )
    # if subspace is none, then we are doing swap based on interchange_dim
    if subspaces is None:
        if mode == "interchange":
            base[..., :interchange_dim] = source[..., :interchange_dim]
        elif mode == "add":
            base[..., :interchange_dim] += source[..., :interchange_dim]
        elif mode == "subtract":
            base[..., :interchange_dim] -= source[..., :interchange_dim]
        elif mode == "collect":
            return base[..., :interchange_dim] # return without side-effect
        return base

    sel_subspace_indices = None
    if use_fast or _can_use_fast(subspaces):
        # its tensor, and each row the same
        if subspace_partition is None:
            sel_subspace_indices = subspaces[0]
        else:
            sel_subspace_indices = []
            for subspace in subspaces[0]:
                sel_subspace_indices.extend(
                    subspace_partition[subspace]
                )
    elif _can_cast_tensor(subspaces):
        sel_subspace_indices = []
        for example_i in range(len(subspaces)):
            # render subspace as column indices
            if subspace_partition is None:
                sel_subspace_indices.append(subspaces[example_i])
            else:
                _sel_subspace_indices = []
                for subspace in subspaces[example_i]:
                    _sel_subspace_indices.extend(
                        subspace_partition[subspace]
                    )
                sel_subspace_indices.append(_sel_subspace_indices)
    
    # _can_use_fast or _can_cast_tensor will prepare the sel_subspace_indices
    if sel_subspace_indices is not None:
        pad_idx = torch.arange(base.shape[-2]).unsqueeze(dim=-1).to(base.device)
        if mode == "interchange":
            base[..., pad_idx, sel_subspace_indices] = source[..., pad_idx, sel_subspace_indices]
        elif mode == "add":
            base[..., pad_idx, sel_subspace_indices] += source[..., pad_idx, sel_subspace_indices]
        elif mode == "subtract":
            base[..., pad_idx, sel_subspace_indices] -= source[..., pad_idx, sel_subspace_indices]
        elif mode == "collect":
            return base[..., pad_idx, sel_subspace_indices] # return without side-effect
    else:
        collect_base = []
        for example_i in range(len(subspaces)):
            # render subspace as column indices
            if subspace_partition is None:
                sel_subspace_indices = subspaces[example_i]
            else:
                sel_subspace_indices = []
                for subspace in subspaces[example_i]:
                    sel_subspace_indices.extend(
                        subspace_partition[subspace]
                    )
            if mode == "interchange":
                base[example_i, ..., sel_subspace_indices] = source[
                    example_i, ..., sel_subspace_indices
                ]
            elif mode == "add":
                base[example_i, ..., sel_subspace_indices] += source[
                    example_i, ..., sel_subspace_indices
                ]
            elif mode == "subtract":
                base[example_i, ..., sel_subspace_indices] -= source[
                    example_i, ..., sel_subspace_indices
                ]
            elif mode == "collect":
                collect_base += [base[example_i, ..., sel_subspace_indices]]
        if mode == "collect":
            return torch.stack(collect_base, dim=0) # return without side-effect

    return base
