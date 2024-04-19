from collections.abc import Sequence
import random, torch, types
import numpy as np
from torch import nn
from .intervenable_modelcard import *
from .interventions import *
from .constants import *

UNIT_LOC_LIST_TYPE = Sequence[int | Sequence | torch.Tensor | None]


def get_internal_model_type(model):
    """Return the model type."""
    return type(model)


def is_stateless(model):
    """Determine if the model is stateful (e.g., rnn) or stateless (e.g.,
    transformer)
    """
    if is_gru(model):
        return False
    return True


def is_gru(model):
    """Determine if this is a transformer model."""
    if (
        type(model) == GRUModel
        or type(model) == GRULMHeadModel
        or type(model) == GRUForClassification
    ):
        return True
    return False


def is_mlp(model):
    """Determine if this is a mlp model."""
    if type(model) == MLPModel or type(model) == MLPForClassification:
        return True
    return False


def is_transformer(model):
    """Determine if this is a transformer model."""
    if not is_gru(model) and not is_mlp(model):
        return True
    return False


def print_forward_hooks(main_module):
    """Function to print forward hooks of a module and its sub-modules."""
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
    """Function to remove all forward and pre-forward hooks from a module and

    its sub-modules.
    """

    # Remove forward hooks
    for _, submodule in main_module.named_modules():
        if hasattr(submodule, "_forward_hooks"):
            hooks = list(submodule._forward_hooks.keys())  # Get a list of hook IDs
            for hook_id in hooks:
                submodule._forward_hooks.pop(hook_id)

        # Remove pre-forward hooks
        if hasattr(submodule, "_forward_pre_hooks"):
            pre_hooks = list(
                submodule._forward_pre_hooks.keys()
            )  # Get a list of pre-hook IDs
            for pre_hook_id in pre_hooks:
                submodule._forward_pre_hooks.pop(pre_hook_id)


def getattr_for_torch_module(model, parameter_name):
    """Recursively fetch the model based on the name."""
    current_module = model
    for param in parameter_name.split("."):
        if "[" in param:
            current_module = getattr(current_module, param.split("[")[0])[
                int(param.split("[")[-1].strip("]"))
            ]
        else:
            current_module = getattr(current_module, param)
    return current_module


def get_dimension_by_component(model_type, model_config, component) -> int | None:
    """Based on the representation, get the aligning dimension size."""

    if component not in type_to_dimension_mapping[model_type]:
        return None

    dimension_proposals = type_to_dimension_mapping[model_type][component]
    for proposal in dimension_proposals:
        if proposal.isnumeric():
            dimension = int(proposal)
        elif "*" in proposal:
            # often constant multiplier with MLP
            dimension = getattr_for_torch_module(
                model_config, proposal.split("*")[0]
            ) * int(proposal.split("*")[1])
        elif "/" in proposal:
            # often split by head number
            if proposal.split("/")[0].isnumeric():
                numr = int(proposal.split("/")[0])
            else:
                numr = getattr_for_torch_module(model_config, proposal.split("/")[0])

            if proposal.split("/")[1].isnumeric():
                denr = int(proposal.split("/")[1])
            else:
                denr = getattr_for_torch_module(model_config, proposal.split("/")[1])
            dimension = int(numr / denr)
        else:
            dimension = getattr_for_torch_module(model_config, proposal)
        if dimension is not None:
            return dimension

    assert False


def get_module_hook(model, representation) -> nn.Module:
    """Render the intervening module with a hook."""
    if (
        get_internal_model_type(model) in type_to_module_mapping
        and representation.component
        in type_to_module_mapping[get_internal_model_type(model)]
    ):
        type_info = type_to_module_mapping[get_internal_model_type(model)][
            representation.component
        ]
        parameter_name = type_info[0]
        hook_type = type_info[1]
        if "%s" in parameter_name and representation.moe_key is None:
            # we assume it is for the layer.
            parameter_name = parameter_name % (representation.layer)
        elif "%s" in parameter_name and representation.moe_key is not None:
            parameter_name = parameter_name % (
                int(representation.layer),
                int(representation.moe_key),
            )
    else:
        parameter_name = ".".join(representation.component.split(".")[:-1])
        if representation.component.split(".")[-1] == "input":
            hook_type = CONST_INPUT_HOOK
        elif representation.component.split(".")[-1] == "output":
            hook_type = CONST_OUTPUT_HOOK

    module = getattr_for_torch_module(model, parameter_name)
    module_hook = getattr(module, hook_type)

    return module_hook


class HandlerList:
    """General class to set hooks and set off hooks."""

    def __init__(self, handlers):
        self.handlers = handlers

    def __len__(self):
        return len(self.handlers)

    def remove(self):
        for handler in self.handlers:
            handler.remove()

    def extend(self, new_handlers):
        self.handlers.extend(new_handlers.handlers)
        return self


def bsd_to_b_sd(tensor):
    """Convert a tensor of shape (b, s, d) to (b, s*d)."""
    if tensor is None:
        return tensor
    b, s, d = tensor.shape
    return tensor.reshape(b, s * d)


def b_sd_to_bsd(tensor, s):
    """Convert a tensor of shape (b, s*d) back to (b, s, d)."""
    if tensor is None:
        return tensor
    b, sd = tensor.shape
    d = sd // s
    return tensor.reshape(b, s, d)


def bhsd_to_bs_hd(tensor):
    """Convert a tensor of shape (b, h, s, d) to (b, s, h*d)."""
    if tensor is None:
        return tensor
    b, h, s, d = tensor.shape
    return tensor.permute(0, 2, 1, 3).reshape(b, s, h * d)


def bs_hd_to_bhsd(tensor, h):
    """Convert a tensor of shape (b, s, h*d) back to (b, h, s, d)."""
    if tensor is None:
        return tensor
    b, s, hd = tensor.shape

    d = hd // h

    return tensor.reshape(b, s, h, d).permute(0, 2, 1, 3)


def output_to_subcomponent(
    output: torch.Tensor, component, model_type, model_config
) -> torch.Tensor:
    """Split the raw output to subcomponents if specified in the config.

    :param output: the original output from the model component.
    :param component: types of model component, such as
    "block_output" and "query_output" or it can be direct referece, such as
    "h[0].mlp.act" which we will not splice into any subcomponent.
    :param model_type: Hugging Face Model Type
    :param model_config: Hugging Face Model Config
    """
    subcomponent = output
    if (
        model_type in type_to_module_mapping
        and component in type_to_module_mapping[model_type]
    ):
        split_last_dim_by = type_to_module_mapping[model_type][component][2:]

        if len(split_last_dim_by) > 2:
            raise ValueError(f"Unsupported {split_last_dim_by}.")

        for i, (split_fn, param) in enumerate(split_last_dim_by):
            if isinstance(param, str):
                param = get_dimension_by_component(model_type, model_config, param)
            subcomponent = split_fn(subcomponent, param)
    return subcomponent


def gather_neurons(
    tensor_input: torch.Tensor, unit, unit_locations_as_list: UNIT_LOC_LIST_TYPE
) -> torch.Tensor:
    """Gather intervening neurons.

    :param tensor_input: tensors of shape (batch_size, sequence_length, ...) if
    `unit` is "pos" or "h", tensors of shape (batch_size, num_heads,
    sequence_length, ...) if `unit` is "h.pos"
    :param unit: the intervention units to gather. Units could be "h" - head
    number, "pos" - position in the sequence, or "dim" - a particular dimension in
    the embedding space. If intervening multiple units, they are ordered and
    separated by `.`. Currently only support "pos", "h", and "h.pos" units.
    :param unit_locations_as_list: tuple of lists of lists of positions to gather
    in tensor_input, according to the unit.
    :return the gathered tensor as tensor_output
    """
    if unit in {"t"}:
        return tensor_input

    if "." in unit:
        unit_locations = (
            torch.tensor(unit_locations_as_list[0], device=tensor_input.device),
            torch.tensor(unit_locations_as_list[1], device=tensor_input.device),
        )
        # we assume unit_locations is a tuple
        head_unit_locations = unit_locations[0]
        pos_unit_locations = unit_locations[1]
        _batch_idx = torch.arange(tensor_input.shape[0])[:, None, None]

        return tensor_input[
            _batch_idx, head_unit_locations[:, :, None], pos_unit_locations[:, None, :]
        ]
    else:
        # For now, when gathering neurons to set, we want to include the entire batch
        # even if we are only intervening on some of them, just so there are no
        # surprising changes in the base shape. I am setting all the None rows
        # to 0 because the scatter function will filter these rows out anyways.
        unit_locations_as_list = [(arr or [0]) for arr in unit_locations_as_list]
        unit_locations = torch.tensor(
            unit_locations_as_list, device=tensor_input.device
        )
        _batch_idx = torch.arange(tensor_input.shape[0])[:, None]
        return tensor_input[_batch_idx, unit_locations]


def scatter_neurons(
    tensor_input: torch.Tensor,
    replacing_tensor_input: torch.Tensor,
    component,
    unit,
    unit_locations_as_list: UNIT_LOC_LIST_TYPE,
    model_type,
    model_config,
    use_fast: bool,
) -> torch.Tensor:
    """Replace selected neurons in `tensor_input` by `replacing_tensor_input`.

    :param tensor_input: tensors of shape (batch_size, sequence_length, ...) if
    `unit` is "pos" or "h", tensors of shape (batch_size, num_heads,
    sequence_length, ...) if `unit` is "h.pos"
    :param replacing_tensor_input: tensors of shape (batch_size, sequence_length,
    ...) if `unit` is "pos" or
    "h", tensors of shape (batch_size, num_heads, sequence_length, ...) if
    `unit` is "h.pos".
    :param component: types of intervention representations, such as
    "block_output" and "query_output"
    :param unit: the intervention units to gather. Units could be "h" - head
    number, "pos" - position in the sequence, or "dim" - a particular dimension in
    the embedding space. If intervening multiple units, they are ordered and
    separated by `.`. Currently only support "pos", "h", and "h.pos" units.
    :param unit_locations_as_list: tuple of lists of lists of positions to gather
    in tensor_input, according to the unit.
    :param model_type: Hugging Face Model Type
    :param model_config: Hugging Face Model Config
    :param use_fast: whether to use fast path (TODO: fast path condition)
    :return the in-place modified tensor_input
    """
    # if tensor is splitted, we need to get the start and end indices
    meta_component = output_to_subcomponent(
        torch.arange(tensor_input.shape[-1]).unsqueeze(dim=0).unsqueeze(dim=0),
        component,
        model_type,
        model_config,
    )

    last_dim = meta_component.shape[-1]

    if "." in unit:
        # extra dimension for multi-level intervention
        unit_locations = (
            torch.tensor(unit_locations_as_list[0], device=tensor_input.device),
            torch.tensor(unit_locations_as_list[1], device=tensor_input.device),
        )

        if unit != "h.pos":
            # TODO: let's leave batch disabling for complex interventions to later
            _batch_idx = torch.arange(tensor_input.shape[0])[:, None, None]
            return tensor_input[
                _batch_idx, unit_locations[0][:, :, None], unit_locations[1][:, None, :]
            ]

        # head-based scattering is only special for transformer-based model
        # replacing_tensor_input: b_s, num_h, s, h_dim -> b_s, s, num_h*h_dim
        old_shape = tensor_input.size()  # b_s, s, x*num_h*d
        new_shape = tensor_input.size()[:-1] + (
            -1,
            meta_component.shape[1],
            last_dim,
        )  # b_s, s, x, num_h, d

        # get whether split by QKV
        # NOTE: type_to_module_mapping[model_type][component][2] is an optional config tuple
        # specifying how to index for a specific component of a single embedding:
        # - the function splitting the embedding vector by component, and
        # - the index of the component within the resulting split.
        if (
            component in type_to_module_mapping[model_type]
            and len(type_to_module_mapping[model_type][component]) > 2
            and type_to_module_mapping[model_type][component][2][0] == split_three
        ):
            _slice_idx = type_to_module_mapping[model_type][component][2][1]
        else:
            _slice_idx = 0

        _batch_idx = torch.arange(tensor_input.shape[0])[:, None, None]
        _head_idx = unit_locations[0][:, :, None]
        _pos_idx = unit_locations[1][:, None, :]
        tensor_permute = tensor_input.view(new_shape).permute(
            0, 3, 1, 2, 4
        )  # b_s, num_h, s, x, d
        tensor_permute[
            _batch_idx,
            _head_idx,
            _pos_idx,
            _slice_idx,
        ] = replacing_tensor_input[:, : _head_idx.shape[1], : _pos_idx.shape[2]]
        # reshape
        tensor_output = tensor_permute.permute(0, 2, 3, 1, 4).view(old_shape)
        return tensor_output  # b_s, s, x*num_h*d

    _batch_idx = torch.tensor(
        [
            i
            for i in range(tensor_input.shape[0])
            if unit_locations_as_list[i] is not None
        ]
    )

    if not len(_batch_idx):
        return tensor_input

    unit_locations = torch.tensor(
        [arr for arr in unit_locations_as_list if arr is not None],
        device=tensor_input.device,
    )

    start_index, end_index = (
        meta_component.min().tolist(),
        (meta_component.max() + 1).tolist(),
    )

    # print(
    #     f"Input shape: {tensor_input.shape}, Replacing shape: {replacing_tensor_input.shape}"
    # )
    # print(
    #     f"Scatter neurons: {_batch_idx}, {unit_locations}, {start_index}, {end_index}"
    # )

    assert (
        unit_locations.shape[0] == _batch_idx.shape[0]
    ), f"unit_locations: {unit_locations.shape}, _batch_idx: {_batch_idx.shape}"

    # in case it is time step, there is no sequence-related index
    if unit in {"t"}:
        # time series models, e.g., gru
        tensor_input[_batch_idx, start_index:end_index] = replacing_tensor_input
        return tensor_input
    elif unit in {"pos"}:
        tensor_input[_batch_idx[:, None], unit_locations, start_index:end_index] = (
            replacing_tensor_input[_batch_idx, :, start_index:end_index]
        )
        return tensor_input
    elif unit in {"h"}:
        # head-based scattering is only special for transformer-based model
        # replacing_tensor_input: b_s, num_h, s, h_dim -> b_s, s, num_h*h_dim
        old_shape = tensor_input.size()  # b_s, s, -1*num_h*d
        new_shape = tensor_input.size()[:-1] + (
            -1,
            meta_component.shape[1],
            last_dim,
        )  # b_s, s, -1, num_h, d
        # get whether split by QKV
        if (
            component in type_to_module_mapping[model_type]
            and len(type_to_module_mapping[model_type][component]) > 2
            and type_to_module_mapping[model_type][component][2][0] == split_three
        ):
            _slice_idx = type_to_module_mapping[model_type][component][2][1]
        else:
            _slice_idx = 0
        tensor_permute = tensor_input.view(new_shape)  # b_s, s, -1, num_h, d
        tensor_permute = tensor_permute.permute(0, 3, 2, 1, 4)  # b_s, num_h, -1, s, d
        tensor_permute[_batch_idx[:, None], unit_locations, _slice_idx] = (
            replacing_tensor_input[_batch_idx]
        )
        # permute back and reshape
        tensor_output = tensor_permute.permute(0, 3, 2, 1, 4)  # b_s, s, -1, num_h, d
        tensor_output = tensor_output.view(old_shape)  # b_s, s, -1*num_h*d
        return tensor_output
    else:
        tensor_input[_batch_idx, unit_locations] = replacing_tensor_input[_batch_idx]
        return tensor_input


def do_intervention(
    base_representation, source_representation, intervention, subspaces
):
    """Do the actual intervention."""

    if isinstance(intervention, types.FunctionType):
        return intervention(base_representation, source_representation)

    num_unit = base_representation.shape[1]

    # flatten
    original_base_shape = base_representation.shape
    if (
        len(original_base_shape) == 2
        or (isinstance(intervention, LocalistRepresentationIntervention))
        or intervention.keep_last_dim
    ):
        # no pos dimension, e.g., gru, or opt-out concate last two dims
        base_representation_f = base_representation
        source_representation_f = source_representation
    elif len(original_base_shape) == 3:
        # b, num_unit (pos), d -> b, num_unit*d
        base_representation_f = bsd_to_b_sd(base_representation)
        source_representation_f = bsd_to_b_sd(source_representation)
    elif len(original_base_shape) == 4:
        # b, num_unit (h), s, d -> b, s, num_unit*d
        base_representation_f = bhsd_to_bs_hd(base_representation)
        source_representation_f = bhsd_to_bs_hd(source_representation)
    else:
        assert False  # what's going on?

    intervened_representation = intervention(
        base_representation_f, source_representation_f, subspaces
    )

    post_d = intervened_representation.shape[-1]

    # unflatten
    if (
        len(original_base_shape) == 2
        or isinstance(intervention, LocalistRepresentationIntervention)
        or intervention.keep_last_dim
    ):
        # no pos dimension, e.g., gru or opt-out concate last two dims
        pass
    elif len(original_base_shape) == 3:
        intervened_representation = b_sd_to_bsd(intervened_representation, num_unit)
    elif len(original_base_shape) == 4:
        intervened_representation = bs_hd_to_bhsd(intervened_representation, num_unit)
    else:
        assert False  # what's going on?

    return intervened_representation


def simple_output_to_subcomponent(output, representation_type, model_config):
    """This is an oversimplied version for demo."""
    return output


def simple_scatter_intervention_output(
    original_output,
    intervened_representation,
    representation_type,
    unit,
    unit_locations,
    model_config,
):
    """This is an oversimplied version for demo."""
    for batch_i, locations in enumerate(unit_locations):
        original_output[batch_i, locations] = intervened_representation[batch_i]


def weighted_average(values, weights):
    if len(values) != len(weights):
        raise ValueError("The length of values and weights must be the same.")

    total = sum(v * w for v, w in zip(values, weights))
    return total / sum(weights)
