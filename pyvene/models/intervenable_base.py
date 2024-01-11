import json, logging
import numpy as np
from collections import OrderedDict
from typing import List, Optional, Tuple, Union, Dict

from pyvene.models.basic_utils import *
from pyvene.models.modeling_utils import *
from pyvene.models.intervention_utils import *
import pyvene.models.interventions
from pyvene.models.constants import CONST_QKV_INDICES
from pyvene.models.configuration_intervenable_model import (
    IntervenableConfig,
    IntervenableRepresentationConfig,
)

from torch import optim
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm, trange


class IntervenableModel(nn.Module):
    """
    Generic intervenable model. Alignments are specified in the config.
    """

    def __init__(self, intervenable_config, model, **kwargs):
        super().__init__()
        self.intervenable_config = intervenable_config
        self.mode = intervenable_config.mode
        intervention_type = intervenable_config.intervenable_interventions_type
        self.is_model_stateless = is_stateless(model)
        self.intervenable_config.intervenable_model_type = type(model) # backfill
        self.use_fast = kwargs["use_fast"] if "use_fast" in kwargs else False
        if self.use_fast:
            logging.warn(
                "Detected use_fast=True means the intervention location "
                "will be static within a batch.\n\nIn case multiple "
                "location tags are passed only the first one will "
                "be considered"
            )
        # each representation can get a different intervention type
        if type(intervention_type) == list:
            assert len(intervention_type) == len(
                intervenable_config.intervenable_representations
            )

        ###
        # We instantiate intervention_layers at locations.
        # Note that the layer name mentioned in the config is
        # abstract. Not the actual module name of the model.
        #
        # This script will automatically convert abstract
        # name into module name if the model type is supported.
        #
        # To support a new model type, you need to provide a
        # mapping between supported abstract type and module name.
        ###
        self.intervenable_representations = {}
        self.interventions = {}
        self._key_collision_counter = {}
        # Flags and counters below are for interventions in the model.generate
        # call. We can intervene on the prompt tokens only, on each generated
        # token, or on a combination of both.
        self._is_generation = False
        self._intervene_on_prompt = None
        self._key_getter_call_counter = {}
        self._key_setter_call_counter = {}
        self._intervention_pointers = {}
        self._intervention_reverse_link = {}

        # hooks are stateful internally, meaning that it's aware of how many times
        # it is called during the execution.
        # TODO: this could be merged with call counter above later.
        self._intervention_state = {}

        # We want to associate interventions with a group to do group-wise interventions.
        self._intervention_group = {}
        _any_group_key = False
        _original_key_order = []
        for i, representation in enumerate(
            intervenable_config.intervenable_representations
        ):
            _key = self._get_representation_key(representation)

            if representation.intervenable_unit not in CONST_VALID_INTERVENABLE_UNIT:
                raise ValueError(
                    f"{representation.intervenable_unit} is not supported as intervenable unit. Valid options: ",
                    f"{CONST_VALID_INTERVENABLE_UNIT}",
                )

            if (
                intervenable_config.intervenable_interventions is not None
                and intervenable_config.intervenable_interventions[0] is not None
            ):
                # we leave this option open but not sure if it is a desired one
                intervention = intervenable_config.intervenable_interventions[i]
            else:
                intervention_function = (
                    intervention_type
                    if type(intervention_type) != list
                    else intervention_type[i]
                )
                intervention = intervention_function(
                    get_intervenable_dimension(
                        get_internal_model_type(model), model.config, representation
                    ),
                    proj_dim=representation.intervenable_low_rank_dimension,
                    # we can partition the subspace, and intervene on subspace
                    subspace_partition=representation.subspace_partition,
                )
                if representation.intervention_link_key in self._intervention_pointers:
                    self._intervention_reverse_link[
                        _key
                    ] = f"link#{representation.intervention_link_key}"
                    intervention = self._intervention_pointers[
                        representation.intervention_link_key
                    ]
                else:
                    intervention = intervention_function(
                        get_intervenable_dimension(
                            get_internal_model_type(model), model.config, representation
                        ),
                        proj_dim=representation.intervenable_low_rank_dimension,
                        # we can partition the subspace, and intervene on subspace
                        subspace_partition=representation.subspace_partition,
                        use_fast=self.use_fast,
                    )
                    # we cache the intervention for sharing if the key is not None
                    if representation.intervention_link_key is not None:
                        self._intervention_pointers[
                            representation.intervention_link_key
                        ] = intervention
                        self._intervention_reverse_link[
                            _key
                        ] = f"link#{representation.intervention_link_key}"

            intervenable_module_hook = get_intervenable_module_hook(
                model, representation
            )
            self.intervenable_representations[_key] = representation
            self.interventions[_key] = (intervention, intervenable_module_hook)
            self._key_getter_call_counter[
                _key
            ] = 0  # we memo how many the hook is called,
            # usually, it's a one time call per
            # hook unless model generates.
            self._key_setter_call_counter[_key] = 0
            self._intervention_state[_key] = InterventionState(_key)
            _original_key_order += [_key]
            if representation.group_key is not None:
                _any_group_key = True
        if self.intervenable_config.sorted_keys is not None:
            logging.warn(
                "The key is provided in the config. "
                "Assuming this is loaded from a pretrained module."
            )
        if (
            self.intervenable_config.sorted_keys is not None
            or "intervenables_sort_fn" not in kwargs
        ):
            self.sorted_intervenable_keys = _original_key_order
        else:
            # the key order is independent of group, it is used to read out intervention locations.
            self.sorted_intervenable_keys = kwargs["intervenables_sort_fn"](
                model, self.intervenable_representations
            )

        # check it follows topological order
        if not check_sorted_intervenables_by_topological_order(
            model, self.intervenable_representations, self.sorted_intervenable_keys
        ):
            raise ValueError(
                "The intervenable_representations in your config must follow the "
                "topological order of model components. E.g., layer 2 intervention "
                "cannot appear before layer 1 in transformers."
            )

        """
        We later use _intervention_group to run actual interventions.
        The order the group by group; and there should not be dependency
        between groups.
        """
        if _any_group_key:
            # In case they are grouped, we would expect the execution order is given
            # by the source inputs.
            _validate_group_keys = []
            for _key in self.sorted_intervenable_keys:
                representation = self.intervenable_representations[_key]
                assert representation.group_key is not None
                if representation.group_key in self._intervention_group:
                    self._intervention_group[representation.group_key].append(_key)
                else:
                    self._intervention_group[representation.group_key] = [_key]
                _validate_group_keys += [representation.group_key]
            for i in range(len(_validate_group_keys) - 1):
                if _validate_group_keys[i] > _validate_group_keys[i + 1]:
                    logging.info(
                        f"This is not a valid group key order: {_validate_group_keys}" 
                    )
                    raise ValueError(
                        "Must be ascending order. "
                        "Interventions would be performed in order within group as well"
                    )
        else:
            # assign each key to an unique group based on topological order
            _group_key_inc = 0
            for _key in self.sorted_intervenable_keys:
                self._intervention_group[_group_key_inc] = [_key]
                _group_key_inc += 1
        # sort group key with ascending order
        self._intervention_group = OrderedDict(sorted(self._intervention_group.items()))

        # cached swap-in activations
        self.activations = {}
        # cached swapped activations (hot)
        self.hot_activations = {}

        # temp fields should not be accessed outside
        self._batched_setter_activation_select = {}
        """
        Activations in the future list is ALWAYS causally before
        the vanilla activation list. This field becomes crucial
        if we intervene at the same place multiple times.
        """
        self.model = model
        self.model_config = model.config
        self.model_type = get_internal_model_type(model)
        self.disable_model_gradients()

    def __str__(self):
        """
        Print out basic info about this intervenable instance
        """
        attr_dict = {
            "model_type": self.model_type,
            "intervenable_interventions_type": self.intervenable_interventions_type,
            "alignabls": self.sorted_intervenable_keys,
            "mode": self.mode,
        }
        return json.dumps(attr_dict, indent=4)

    def _get_representation_key(self, representation):
        """
        Provide unique key for each intervention
        """
        l = representation.intervenable_layer
        r = representation.intervenable_representation_type
        u = representation.intervenable_unit
        n = representation.max_number_of_units
        key_proposal = f"layer.{l}.repr.{r}.unit.{u}.nunit.{n}"
        if key_proposal not in self._key_collision_counter:
            self._key_collision_counter[key_proposal] = 0
        else:
            self._key_collision_counter[key_proposal] += 1
        return f"{key_proposal}#{self._key_collision_counter[key_proposal]}"

    def _reset_hook_count(self):
        """
        Reset the hook count before any generate call
        """
        self._key_getter_call_counter = dict.fromkeys(self._key_getter_call_counter, 0)
        self._key_setter_call_counter = dict.fromkeys(self._key_setter_call_counter, 0)
        for k, _ in self._intervention_state.items():
            self._intervention_state[k].reset()

    def _remove_forward_hooks(self):
        """
        Clean up all the remaining hooks before any call
        """
        remove_forward_hooks(self.model)

    def _cleanup_states(self, skip_activation_gc=False):
        """
        Clean up all old in memo states of interventions
        """
        self._is_generation = False
        self._remove_forward_hooks()
        self._reset_hook_count()
        if not skip_activation_gc:
            self.activations.clear()
            self.hot_activations.clear()
            self._batched_setter_activation_select.clear()
        else:
            self.activations = {}
            self.hot_activations = {}
            self._batched_setter_activation_select = {}

    def get_trainable_parameters(self):
        """
        Return trainable params as key value pairs
        """
        ret_params = []
        for k, v in self.interventions.items():
            if isinstance(v[0], pyvene.models.interventions.TrainableIntervention):
                ret_params += [p for p in v[0].parameters()]
        return ret_params

    def get_cached_activations(self):
        """
        Return the cached activations with keys
        """
        return self.activations

    def get_cached_hot_activations(self):
        """
        Return the cached hot activations with linked keys
        """
        return self.hot_activations

    def set_temperature(self, temp: torch.Tensor):
        """
        Set temperature if needed
        """
        for k, v in self.interventions.items():
            if isinstance(v[0], pyvene.models.interventions.BoundlessRotatedSpaceIntervention):
                v[0].set_temperature(temp)

    def disable_model_gradients(self):
        """
        Disable gradient in the model
        """
        # Freeze all model weights
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def disable_intervention_gradients(self):
        """
        Disable gradient in the trainable intervention
        """
        # Freeze all intervention weights
        pass

    def set_device(self, device):
        """
        Set device of interventions and the model
        """
        for k, v in self.interventions.items():
            if isinstance(v[0], pyvene.models.interventions.TrainableIntervention):
                v[0].to(device)
        self.model.to(device)

    def get_device(self):
        """
        Get device of interventions and the model
        """
        return self.model.device

    def count_parameters(self):
        """
        Set device of interventions and the model
        """
        _linked_key_set = set([])
        total_parameters = 0
        for k, v in self.interventions.items():
            if isinstance(v[0], pyvene.models.interventions.TrainableIntervention):
                if k in self._intervention_reverse_link:
                    if not self._intervention_reverse_link[k] in _linked_key_set:
                        _linked_key_set.add(self._intervention_reverse_link[k])
                        total_parameters += count_parameters(v[0])
                else:
                    total_parameters += count_parameters(v[0])
        return total_parameters

    def set_zero_grad(self):
        """
        Set device of interventions and the model
        """
        for k, v in self.interventions.items():
            if isinstance(v[0], pyvene.models.interventions.TrainableIntervention):
                v[0].zero_grad()

    def save(
        self, save_directory, save_to_hf_hub=False, hf_repo_name="my-awesome-model"
    ):
        """
        Save interventions to disk or hub
        """
        if save_to_hf_hub:
            from huggingface_hub import HfApi

            api = HfApi()

        create_directory(save_directory)

        saving_config = copy.deepcopy(self.intervenable_config)
        saving_config.sorted_keys = self.sorted_intervenable_keys
        saving_config.intervenable_model_type = str(
            saving_config.intervenable_model_type
        )
        saving_config.intervenable_interventions_type = []
        saving_config.intervention_dimensions = []
        for k, v in self.interventions.items():
            intervention = v[0]
            saving_config.intervenable_interventions_type += [str(type(intervention))]
            binary_filename = f"intkey_{k}.bin"
            # save intervention binary file
            if isinstance(intervention, pyvene.models.interventions.TrainableIntervention):
                logging.warn(f"Saving trainable intervention to {binary_filename}.")
                torch.save(
                    intervention.state_dict(),
                    os.path.join(save_directory, binary_filename),
                )
                if save_to_hf_hub:
                    # push to huggingface hub
                    try:
                        api.create_repo(hf_repo_name)
                    except:
                        logging.warn(
                            f"Skipping creating the repo since "
                            f"either {hf_repo_name} exists or having authentication error."
                        )
                    api.upload_file(
                        path_or_fileobj=os.path.join(save_directory, binary_filename),
                        path_in_repo=binary_filename,
                        repo_id=hf_repo_name,
                        repo_type="model",
                    )
            saving_config.intervention_dimensions += [intervention.interchange_dim]
        # save metadata config
        saving_config.save_pretrained(save_directory)
        if save_to_hf_hub:
            # push to huggingface hub
            try:
                api.create_repo(hf_repo_name)
            except:
                logging.warn(
                    f"Skipping creating the repo since "
                    f"either {hf_repo_name} exists or having authentication error."
                )
            api.upload_file(
                path_or_fileobj=os.path.join(save_directory, "config.json"),
                path_in_repo="config.json",
                repo_id=hf_repo_name,
                repo_type="model",
            )

    @staticmethod
    def load(load_directory, model, local_directory=None, from_huggingface_hub=False):
        """
        Load interventions from disk or hub
        """
        if not os.path.exists(load_directory) or from_huggingface_hub:
            if local_directory is None:
                raise ValueError(
                    "You have to provide local_directory to save hf files."
                )
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id=load_directory,
                filename="config.json",
                cache_dir=local_directory,
            )
            # simple overwrite
            load_directory = local_directory

        # load config
        saving_config = IntervenableConfig.from_pretrained(load_directory)
        saving_config.intervenable_model_type = get_type_from_string(
            saving_config.intervenable_model_type
        )
        if not isinstance(model, saving_config.intervenable_model_type):
            raise ValueError(
                f"model type {str(type(model))} is not "
                f"matching with {str(saving_config.intervenable_model_type)}"
            )
        casted_intervenable_interventions_type = []
        for type_str in saving_config.intervenable_interventions_type:
            casted_intervenable_interventions_type += [get_type_from_string(type_str)]
        saving_config.intervenable_interventions_type = (
            casted_intervenable_interventions_type
        )
        casted_intervenable_representations = []
        for (
            intervenable_representation_opts
        ) in saving_config.intervenable_representations:
            casted_intervenable_representations += [
                IntervenableRepresentationConfig(*intervenable_representation_opts)
            ]
        saving_config.intervenable_representations = casted_intervenable_representations
        intervenable = IntervenableModel(saving_config, model)

        # load binary files
        for i, (k, v) in enumerate(intervenable.interventions.items()):
            intervention = v[0]
            binary_filename = f"intkey_{k}.bin"
            if isinstance(intervention, pyvene.models.interventions.TrainableIntervention):
                if not os.path.exists(load_directory) or from_huggingface_hub:
                    hf_hub_download(
                        repo_id=load_directory,
                        filename=binary_filename,
                        cache_dir=local_directory,
                    )
                logging.warn(f"Loading trainable intervention from {binary_filename}.")
                intervention.load_state_dict(
                    torch.load(os.path.join(load_directory, binary_filename))
                )
            intervention.interchange_dim = saving_config.intervention_dimensions[i]

        return intervenable

    def _gather_intervention_output(
        self, output, intervenable_representations_key, unit_locations
    ) -> torch.Tensor:
        """
        Gather intervening activations from the output based on indices
        """

        if (
            intervenable_representations_key in self._intervention_reverse_link
            and self._intervention_reverse_link[intervenable_representations_key]
            in self.hot_activations
        ):
            # hot gather
            # clone is needed here by acting as a different module
            # to avoid gradient conflict.
            #
            # enable the following line when an error is hit
            # torch.autograd.set_detect_anomaly(True)
            selected_output = self.hot_activations[
                self._intervention_reverse_link[intervenable_representations_key]
            ]
        else:
            # cold gather
            original_output = output
            # data structure casting
            if isinstance(output, tuple):
                original_output = output[0]
            # gather subcomponent
            original_output = self._output_to_subcomponent(
                original_output, intervenable_representations_key
            )

            # gather based on intervention locations
            selected_output = gather_neurons(
                original_output,
                self.intervenable_representations[
                    intervenable_representations_key
                ].intervenable_unit,
                unit_locations,
            )

        return selected_output

    def _output_to_subcomponent(
        self,
        output,
        intervenable_representations_key,
    ) -> List[torch.Tensor]:
        """
        Helps to get subcomponent of inputs/outputs of a hook

        For instance, we need to separate QKV from a hidden representation
        by slicing the original output
        """
        return output_to_subcomponent(
            output,
            self.intervenable_representations[
                intervenable_representations_key
            ].intervenable_representation_type,
            self.model_type,
            self.model_config,
        )

    def _scatter_intervention_output(
        self,
        output,
        intervened_representation,
        intervenable_representations_key,
        unit_locations,
    ) -> torch.Tensor:
        """
        Scatter in the intervened activations in the output
        """
        original_output = output
        # data structure casting
        if isinstance(output, tuple):
            original_output = output[0]

        intervenable_representation_type = self.intervenable_representations[
            intervenable_representations_key
        ].intervenable_representation_type
        intervenable_unit = self.intervenable_representations[
            intervenable_representations_key
        ].intervenable_unit

        replaced_output = scatter_neurons(
            original_output,
            intervened_representation,
            intervenable_representation_type,
            intervenable_unit,
            unit_locations,
            self.model_type,
            self.model_config,
            self.use_fast,
        )
        return replaced_output

    def _intervention_getter(
        self,
        intervenable_keys,
        unit_locations,
    ) -> HandlerList:
        """
        Create a list of getter handlers that will fetch activations
        """
        handlers = []
        for key_i, key in enumerate(intervenable_keys):
            intervention, intervenable_module_hook = self.interventions[key]

            def hook_callback(model, args, kwargs, output=None):
                if self._is_generation:
                    is_prompt = self._key_getter_call_counter[key] == 0
                    if not self._intervene_on_prompt or is_prompt:
                        self._key_getter_call_counter[key] += 1
                    if self._intervene_on_prompt ^ is_prompt:
                        return  # no-op
                if output is None:
                    if len(args) == 0:  # kwargs based calls
                        # PR: https://github.com/frankaging/align-transformers/issues/11
                        # We cannot assume the dict only contain one element
                        output = kwargs[list(kwargs.keys())[0]]
                    else:
                        output = args

                if isinstance(intervention, pyvene.models.interventions.SkipIntervention):
                    selected_output = self._gather_intervention_output(
                        args[0],  # this is actually the input to the module
                        key,
                        unit_locations[key_i],
                    )
                else:
                    selected_output = self._gather_intervention_output(
                        output, key, unit_locations[key_i]
                    )

                if self.is_model_stateless:
                    # WARNING: might be worth to check the below assertion at runtime,
                    # but commenting it out for now just to avoid confusion.
                    # assert key not in self.activations
                    self.activations[key] = selected_output
                else:
                    state_select_flag = []
                    for unit_location in unit_locations[key_i]:
                        if (
                            self._intervention_state[key].getter_version()
                            in unit_location
                        ):
                            state_select_flag += [True]
                        else:
                            state_select_flag += [False]
                    # for stateful model (e.g., gru), we save extra activations and metadata to do
                    # stateful interventions.
                    self.activations.setdefault(key, []).append(
                        (selected_output, state_select_flag)
                    )
                # set version for stateful models
                self._intervention_state[key].inc_getter_version()

            handlers.append(intervenable_module_hook(hook_callback, with_kwargs=True))

        return HandlerList(handlers)

    def _tidy_stateful_activations(
        self,
    ):
        _need_tidify = False
        for _, v in self.activations.items():
            if isinstance(v[0], tuple) and isinstance(v[0][1], list):
                _need_tidify = True
                break
        if _need_tidify:
            for k, v in self.activations.items():
                self._tidify_activations = [[] for _ in range(v[0][0].shape[0])]
                for t in range(len(v)):
                    activations_at_t = v[t][0]  # a batched tensor
                    states_at_t = (
                        torch.tensor(v[t][1]).bool().to(activations_at_t.device)
                    )  # a batched bools
                    selected_activations = activations_at_t[states_at_t]
                    selected_indices = torch.nonzero(states_at_t).squeeze()
                    if len(selected_indices.shape) == 0:
                        selected_indices = selected_indices.unsqueeze(0)
                    for index, activation in zip(
                        selected_indices, selected_activations
                    ):
                        self._tidify_activations[index].append(activation)
                self.activations[k] = self._tidify_activations

    def _reconcile_stateful_cached_activations(
        self,
        key,
        intervening_activations,
        intervening_unit_locations,
    ):
        """Based on the key, we consolidate activations based on key's state"""
        cached_activations = self.activations[key]
        if self.is_model_stateless:
            # nothing to reconcile if stateless
            return cached_activations

        state_select_flag = []
        for unit_location in intervening_unit_locations:
            if self._intervention_state[key].setter_version() in unit_location:
                state_select_flag += [True]
            else:
                state_select_flag += [False]
        state_select_flag = (
            torch.tensor(state_select_flag).bool().to(intervening_activations.device)
        )
        selected_indices = torch.nonzero(state_select_flag).squeeze()
        if len(selected_indices.shape) == 0:
            selected_indices = selected_indices.unsqueeze(0)

        # fill activations with proposed only source activations
        reconciled_activations = []
        for index, select_version in enumerate(
            self._batched_setter_activation_select[key]
        ):
            if index in selected_indices:
                reconciled_activations += [cached_activations[index][select_version]]
            else:
                # WARNING: put a dummy tensor, super danger here but let's trust the code for now.
                reconciled_activations += [
                    torch.zeros_like(cached_activations[index][0])
                ]
        # increment pointer for those we are actually intervening
        for index in selected_indices:
            self._batched_setter_activation_select[key][index] += 1
        # for non-intervening ones, we copy again from base
        reconciled_activations = torch.stack(reconciled_activations, dim=0)  # batched
        # reconciled_activations[~state_select_flag] = intervening_activations[~state_select_flag]

        return reconciled_activations

    def _intervention_setter(
        self,
        intervenable_keys,
        unit_locations_base,
        subspaces,
    ) -> HandlerList:
        """
        Create a list of setter handlers that will set activations
        """
        self._tidy_stateful_activations()

        handlers = []
        for key_i, key in enumerate(intervenable_keys):
            intervention, intervenable_module_hook = self.interventions[key]
            self._batched_setter_activation_select[key] = [
                0 for _ in range(len(unit_locations_base[0]))
            ]  # batch_size

            def hook_callback(model, args, kwargs, output=None):
                if self._is_generation:
                    is_prompt = self._key_setter_call_counter[key] == 0
                    if not self._intervene_on_prompt or is_prompt:
                        self._key_setter_call_counter[key] += 1
                    if self._intervene_on_prompt ^ is_prompt:
                        return  # no-op
                if output is None:
                    if len(args) == 0:  # kwargs based calls
                        # PR: https://github.com/frankaging/align-transformers/issues/11
                        # We cannot assume the dict only contain one element
                        output = kwargs[list(kwargs.keys())[0]]
                    else:
                        output = args

                selected_output = self._gather_intervention_output(
                    output, key, unit_locations_base[key_i]
                )
                # TODO: need to figure out why clone is needed
                if not self.is_model_stateless:
                    selected_output = selected_output.clone()

                intervened_representation = do_intervention(
                    selected_output,
                    self._reconcile_stateful_cached_activations(
                        key,
                        selected_output,
                        unit_locations_base[key_i],
                    ),
                    intervention,
                    subspaces[key_i] if subspaces is not None else None,
                )
                # setter can produce hot activations for shared subspace interventions if linked
                if key in self._intervention_reverse_link:
                    self.hot_activations[
                        self._intervention_reverse_link[key]
                    ] = intervened_representation.clone()
                # patched in the intervned activations
                output = self._scatter_intervention_output(
                    output, intervened_representation, key, unit_locations_base[key_i]
                )
                # set version for stateful models
                self._intervention_state[key].inc_setter_version()

            handlers.append(intervenable_module_hook(hook_callback, with_kwargs=True))

        return HandlerList(handlers)

    def _input_validation(
        self,
        base,
        sources,
        unit_locations,
        activations_sources,
        subspaces,
    ):
        """Fail fast input validation"""
        if self.mode == "parallel":
            assert "sources->base" in unit_locations
        elif activations_sources is None and self.mode == "serial":
            assert "sources->base" not in unit_locations

        # sources may contain None, but length should match
        if sources is not None:
            if len(sources) != len(self._intervention_group):
                raise ValueError(
                    f"Source length {len(sources)} is not "
                    f"equal to intervention length {len(self._intervention_group)}."
                )
        else:
            if len(activations_sources) != len(self._intervention_group):
                raise ValueError(
                    f"Source activations length {len(activations_sources)} is not "
                    f"equal to intervention length {len(self._intervention_group)}."
                )

        # if it is stateful models, the passed in activations need to have states
        if not self.is_model_stateless and activations_sources is not None:
            for _, v in activations_sources.items():
                if (
                    isinstance(v, list)
                    and isinstance(v[0], tuple)
                    and isinstance(v[0][1], list) != True
                ):
                    raise ValueError(
                        f"Stateful models need nested activations. See our documentions."
                    )

    def _output_validation(
        self,
    ):
        """Safe guarding the execution by checking memory states"""
        if self.is_model_stateless:
            for k, v in self._intervention_state.items():
                if v.getter_version() > 1 or v.setter_version() > 1:
                    raise Exception(
                        f"For stateless model, each getter and setter "
                        f"should be called only once: {self._intervention_state}"
                    )

    def _flatten_input_dict_as_batch(self, input_dict):
        # we also accept grouped sources, will batch them and provide partition info.
        if not isinstance(input_dict, dict):
            assert isinstance(input_dict, list)
            flatten_input_dict = {}
            for k, v in input_dict[0].items():
                flatten_input_dict[k] = {}
            for i in range(0, len(input_dict)):
                for k, v in input_dict[i].items():
                    flatten_input_dict[k] += [v]
            for k, v in flatten_input_dict.items():
                # flatten as one single batch
                flatten_input_dict[k] = torch.cat(v, dim=0)
        else:
            flatten_input_dict = input_dict
        return flatten_input_dict

    def _get_partition_size(self, input_dict):
        if not isinstance(input_dict, dict):
            assert isinstance(input_dict, list)
            return len(input_dict)
        else:
            return 1

    def _wait_for_forward_with_parallel_intervention(
        self,
        sources,
        unit_locations,
        activations_sources: Optional[Dict] = None,
        subspaces: Optional[List] = None,
    ):
        torch.autograd.set_detect_anomaly(True)
        all_set_handlers = HandlerList([])
        unit_locations_sources = unit_locations["sources->base"][0]
        unit_locations_base = unit_locations["sources->base"][1]
        # for each source, we hook in getters to cache activations
        # at each aligning representations
        if activations_sources is None:
            assert len(sources) == len(self._intervention_group)
            for group_id, intervenable_keys in self._intervention_group.items():
                if sources[group_id] is None:
                    continue  # smart jump for advance usage only
                group_get_handlers = HandlerList([])
                for intervenable_key in intervenable_keys:
                    get_handlers = self._intervention_getter(
                        [intervenable_key],
                        [
                            unit_locations_sources[
                                self.sorted_intervenable_keys.index(intervenable_key)
                            ]
                        ],
                    )
                    group_get_handlers.extend(get_handlers)
                _ = self.model(**sources[group_id])
                group_get_handlers.remove()
        else:
            # simply patch in the ones passed in
            self.activations = activations_sources
            for _, passed_in_intervenable_key in enumerate(self.activations):
                assert passed_in_intervenable_key in self.sorted_intervenable_keys

        # in parallel mode, we swap cached activations all into
        # base at once
        for group_id, intervenable_keys in self._intervention_group.items():
            for intervenable_key in intervenable_keys:
                # skip in case smart jump
                if intervenable_key in self.activations:
                    set_handlers = self._intervention_setter(
                        [intervenable_key],
                        [
                            unit_locations_base[
                                self.sorted_intervenable_keys.index(intervenable_key)
                            ]
                        ],
                        # assume same group targeting the same subspace
                        [
                            subspaces[
                                self.sorted_intervenable_keys.index(intervenable_key)
                            ]
                        ]
                        if subspaces is not None
                        else None,
                    )
                    # for setters, we don't remove them.
                    all_set_handlers.extend(set_handlers)
        return all_set_handlers

    def _wait_for_forward_with_serial_intervention(
        self,
        sources,
        unit_locations,
        activations_sources: Optional[Dict] = None,
        subspaces: Optional[List] = None,
    ):
        all_set_handlers = HandlerList([])
        for group_id, intervenable_keys in self._intervention_group.items():
            if sources[group_id] is None:
                continue  # smart jump for advance usage only
            for intervenable_key_id, intervenable_key in enumerate(intervenable_keys):
                if group_id != len(self._intervention_group) - 1:
                    unit_locations_key = f"source_{group_id}->source_{group_id+1}"
                else:
                    unit_locations_key = f"source_{group_id}->base"
                unit_locations_source = unit_locations[unit_locations_key][0][
                    intervenable_key_id
                ]
                if unit_locations_source is None:
                    continue  # smart jump for advance usage only

                unit_locations_base = unit_locations[unit_locations_key][1][
                    intervenable_key_id
                ]
                if activations_sources is None:
                    # get activation from source_i
                    get_handlers = self._intervention_getter(
                        [intervenable_key],
                        [unit_locations_source],
                    )
                else:
                    self.activations[intervenable_key] = activations_sources[
                        intervenable_key
                    ]
            # call once per group. each intervention is by its own group by default
            if activations_sources is None:
                # this is when previous setter and THEN the getter get called
                _ = self.model(**sources[group_id])
                get_handlers.remove()
                # remove existing setters after getting the curr intervened reprs
                if len(all_set_handlers) > 0:
                    all_set_handlers.remove()
                    all_set_handlers = HandlerList([])

            for intervenable_key in intervenable_keys:
                # skip in case smart jump
                if intervenable_key in self.activations:
                    # set with intervened activation to source_i+1
                    set_handlers = self._intervention_setter(
                        [intervenable_key],
                        [unit_locations_base],
                        # assume the order
                        [
                            subspaces[
                                self.sorted_intervenable_keys.index(intervenable_key)
                            ]
                        ]
                        if subspaces is not None
                        else None,
                    )
                    # for setters, we don't remove them.
                    all_set_handlers.extend(set_handlers)
        return all_set_handlers

    def forward(
        self,
        base,
        sources: Optional[List] = None,
        unit_locations: Optional[Dict] = None,
        activations_sources: Optional[Dict] = None,
        subspaces: Optional[List] = None,
    ):
        """
        Main forward function that serves a wrapper to
        actual model forward calls. It will use forward
        hooks to do interventions.

        In essense, sources will lead to getter hooks to
        get activations. We will use these activations to
        intervene on our base example.

        Parameters:
        base:                The base example.
        sources:             A list of source examples.
        unit_locations:      The intervention locations.
        activations_sources: A list of representations.
        subspace:            Subspace interventions.

        Return:
        base_output: the non-intervened output of the base
        input.
        counterfactual_outputs: the intervened output of the
        base input.

        Notes:

        1) unit_locations
        unit_locations is a dict where keys are tied with
        example pairs involved in one intervention as,
        {
            "sources->base" : List[]
        }

        the shape can be

        2 * num_intervention * bs * num_max_unit

        OR

        2 * num_intervention * num_intervention_level * bs * num_max_unit

        if we intervene on h.pos which is a nested intervention location.

        2) subspaces
        subspaces is a list of indices indicating which subspace will
        this intervention target given an example in the batch.

        An intervention could be initialized with subspace parition as,
        [[... subspace_1 ...], [... subspace_2 ...], [rest]]

        An intervention may be targeting a specific partition.

        This input field should look like something like,
        [
            [[subspace indices], [subspace indices]], <- for the first intervention
            None,                                     <- for the second intervention
            [[subspace indices], [subspace indices]]
        ]

        Only setter (where do_intervention is called) needs this field.

        *We assume base and source targetting the same subspace for now.
        *We assume only a single space is targeted for now (although 2d list is provided).

        Since we now support group-based intervention, the number of sources
        should be equal to the total number of groups.
        """
        self._cleanup_states()

        # if no source inputs, we are calling a simple forward
        if sources is None and activations_sources is None:
            return self.model(**base), None

        self._input_validation(
            base,
            sources,
            unit_locations,
            activations_sources,
            subspaces,
        )

        # returning un-intervened output without gradients
        with torch.inference_mode():
            base_outputs = self.model(**base)

        try:
            # intervene
            if self.mode == "parallel":
                set_handlers_to_remove = (
                    self._wait_for_forward_with_parallel_intervention(
                        sources,
                        unit_locations,
                        activations_sources,
                        subspaces,
                    )
                )
            elif self.mode == "serial":
                set_handlers_to_remove = (
                    self._wait_for_forward_with_serial_intervention(
                        sources,
                        unit_locations,
                        activations_sources,
                        subspaces,
                    )
                )

            # run intervened forward
            counterfactual_outputs = self.model(**base)
            set_handlers_to_remove.remove()

            self._output_validation()
        except Exception as e:
            raise e
        finally:
            self._cleanup_states(
                skip_activation_gc=sources is None and activations_sources is not None
            )

        return base_outputs, counterfactual_outputs

    def generate(
        self,
        base,
        sources: Optional[List] = None,
        unit_locations: Optional[Dict] = None,
        activations_sources: Optional[Dict] = None,
        intervene_on_prompt: bool = True,
        subspaces: Optional[List] = None,
        **kwargs,
    ):
        """
        Intervenable generation function that serves a
        wrapper to regular model generate calls.

        Currently, we support basic interventions **in the
        prompt only**. We will support generation interventions
        in the next release.

        TODO: Unroll sources and intervene in the generation step.

        Parameters:
        base:                The base example.
        sources:             A list of source examples.
        unit_locations:      The intervention locations of
                             base.
        activations_sources: A list of representations.
        intervene_on_prompt: Whether only intervene on prompt.
        **kwargs:            All other generation parameters.

        Return:
        base_output: the non-intervened output of the base
        input.
        counterfactual_outputs: the intervened output of the
        base input.
        """
        self._cleanup_states()

        self._intervene_on_prompt = intervene_on_prompt
        self._is_generation = True

        if sources is None and activations_sources is None:
            return self.model.generate(inputs=base["input_ids"], **kwargs), None

        self._input_validation(
            base,
            sources,
            unit_locations,
            activations_sources,
            subspaces,
        )

        # returning un-intervened output without gradients
        with torch.inference_mode():
            base_outputs = self.model.generate(inputs=base["input_ids"], **kwargs)

        set_handlers_to_remove = None
        try:
            # intervene
            if self.mode == "parallel":
                set_handlers_to_remove = (
                    self._wait_for_forward_with_parallel_intervention(
                        sources,
                        unit_locations,
                        activations_sources,
                        subspaces,
                    )
                )
            elif self.mode == "serial":
                set_handlers_to_remove = (
                    self._wait_for_forward_with_serial_intervention(
                        sources,
                        unit_locations,
                        activations_sources,
                        subspaces,
                    )
                )

            # run intervened generate
            counterfactual_outputs = self.model.generate(
                inputs=base["input_ids"], **kwargs
            )
        except Exception as e:
            raise e
        finally:
            if set_handlers_to_remove is not None:
                set_handlers_to_remove.remove()
            self._is_generation = False
            self._cleanup_states(
                skip_activation_gc=sources is None and activations_sources is not None
            )

        return base_outputs, counterfactual_outputs

    def _batch_process_unit_location(self, inputs):
        """
        Convert original data batch according
        to the intervenable settings.

        The function respects inputs in the following
        data format.


        Each location list in the raw input as,

        [[i, j, ...], [m, n, ...], ...] batched
        where i, j are the unit index, the outter
        list is for the batch


        Possible fields in the input:

        inputs["source_0->base.0.pos"] -> batched
        inputs["source_0->base.1.pos"] -> batched
        AND
        inputs["source_0->source_1.0.pos"] -> batched
        inputs["source_0->source_1.1.pos"] -> batched
        ...

        multiple source locations are included in case
        there are multiple sources.

        We also need to consider whether we are doing
        parallel or serial interventions.

        We also need to consider the granularity. In case
        we are intervening h.pos, which is a specific location
        in a specific head:

        inputs["source_0->base.0.pos"] -> batched
        inputs["source_0->source_1.0.h"] -> batched

        inputs["source_0->base.0.pos"] -> batched
        inputs["source_0->source_1.0.pos"] -> batched
        """
        batched_location_dict = {}

        _source_ind = []
        for k, _ in inputs.items():
            if "->" in k:
                for sub_k in k.split("->"):
                    if "source" in sub_k:
                        _source_ind += [int(sub_k.split("_")[1])]
        _max_source_ind = max(_source_ind)

        # we assume source_0 -> source_1, ..., source_last -> base
        # each pair uses an intervention

        if self.mode == "parallel":
            # all source into base at once but may engage different locations
            _curr_source_ind = 0
            _parallel_aggr_left = []
            _parallel_aggr_right = []
            for _, rep in self.intervenable_representations.items():
                _curr_source_ind_inc = _curr_source_ind + 1
                _prefix = f"source_{_curr_source_ind}->base"
                _prefix_left = f"{_prefix}.0"
                _prefix_right = f"{_prefix}.1"
                _sub_loc_aggr_left = []  # 3d
                _sub_loc_aggr_right = []  # 3d
                for sub_loc in rep.intervenable_unit.split("."):
                    _sub_loc_aggr_left += [inputs[f"{_prefix_left}.{sub_loc}"]]
                    _sub_loc_aggr_right += [inputs[f"{_prefix_right}.{sub_loc}"]]
                if len(rep.intervenable_unit.split(".")) == 1:
                    _sub_loc_aggr_left = _sub_loc_aggr_left[0]
                    _sub_loc_aggr_right = _sub_loc_aggr_right[0]
                _parallel_aggr_left += [_sub_loc_aggr_left]  # 3D or 4D
                _parallel_aggr_right += [_sub_loc_aggr_right]  # 3D or 4D
                _curr_source_ind += 1

            batched_location_dict["sources->base"] = (
                _parallel_aggr_left,
                _parallel_aggr_right,
            )

        else:
            # source into another source and finally to the base engaging different locations
            _curr_source_ind = 0
            for _, rep in self.intervenable_representations.items():
                _curr_source_ind_inc = _curr_source_ind + 1
                _prefix = (
                    f"source_{_curr_source_ind}->base"
                    if _curr_source_ind + 1 == len(self.intervenable_representations)
                    else f"source_{_curr_source_ind}->source{_curr_source_ind_inc}"
                )
                _prefix_left = f"{_prefix}.0"
                _prefix_right = f"{_prefix}.1"
                _sub_loc_aggr_left = []  # 3d
                _sub_loc_aggr_right = []  # 3d
                for sub_loc in rep.intervenable_unit.split("."):
                    _sub_loc_aggr_left += [inputs[f"{_prefix_left}.{sub_loc}"]]
                    _sub_loc_aggr_right += [inputs[f"{_prefix_right}.{sub_loc}"]]
                if len(rep.intervenable_unit.split(".")) == 1:
                    _sub_loc_aggr_left = _sub_loc_aggr_left[0]
                    _sub_loc_aggr_right = _sub_loc_aggr_right[0]
                _curr_source_ind += 1
                batched_location_dict[_prefix] = (
                    [_sub_loc_aggr_left],  # 3D or 4D
                    [_sub_loc_aggr_right],  # 3D or 4D
                )

        return batched_location_dict

    def train(
        self,
        train_dataloader,
        compute_loss,
        compute_metrics,
        inputs_collator,
        **kwargs,
    ):
        """
        The method find alignment.

        a.k.a. training the intervention

        Notes:
        1) we use Adam, and linear lr scheduling.
        2) you can pass in lr or using default 1e-3
        """
        # preprocess basic kwargs
        lr = kwargs["lr"] if "lr" in kwargs else 1e-3
        epochs = kwargs["epochs"] if "epochs" in kwargs else 10
        warm_up_steps = kwargs["warm_up_steps"] if "warm_up_steps" in kwargs else 0.1
        gradient_accumulation_steps = (
            kwargs["gradient_accumulation_steps"]
            if "gradient_accumulation_steps" in kwargs
            else 1
        )

        # some deeper kwargs
        t_total = int(len(train_dataloader) * epochs)
        warm_up_steps = 0.1 * t_total
        target_total_step = len(train_dataloader) * epochs
        optimizer_params = [{"params": self.get_trainable_parameters()}]
        optimizer = (
            kwargs["optimizer"]
            if "optimizer" in kwargs
            else optim.Adam(optimizer_params, lr=lr)
        )
        scheduler = (
            kwargs["scheduler"]
            if "scheduler" in kwargs
            else get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warm_up_steps, num_training_steps=t_total
            )
        )

        # in case we need additional temp scheduling
        temperature_start = 50.0
        temperature_end = 0.1
        temperature_schedule = (
            torch.linspace(temperature_start, temperature_end, target_total_step)
            .to(torch.bfloat16)
            .to(self.get_device())
        )

        # train main loop
        remove_forward_hooks(self.model)
        self.model.eval()  # train enables drop-off but no grads
        epoch_iterator = trange(0, int(epochs), desc="Epoch")
        total_step = 0
        for epoch in epoch_iterator:
            for step, inputs in enumerate(train_dataloader):
                if inputs_collator is not None:
                    inputs = inputs_collator(inputs)
                b_s = inputs["input_ids"].shape[0]
                unit_location_dict = self._batch_process_unit_location(inputs)
                _, counterfactual_outputs = self(
                    {"input_ids": inputs["input_ids"]},
                    [{"input_ids": inputs["source_input_ids"]}],
                    unit_location_dict,
                    subspaces=inputs["subspaces"] if "subspaces" in inputs else None,
                )
                eval_metrics = compute_metrics(
                    [counterfactual_outputs.logits], [inputs["labels"]]
                )

                # loss and backprop
                loss = compute_loss(counterfactual_outputs.logits, inputs["labels"])
                loss_str = round(loss.item(), 2)
                epoch_iterator.set_postfix({"loss": loss_str, "acc": eval_metrics})

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                loss.backward()
                if total_step % gradient_accumulation_steps == 0:
                    if not (gradient_accumulation_steps > 1 and total_step == 0):
                        optimizer.step()
                        scheduler.step()
                        self.set_zero_grad()
                        self.set_temperature(temperature_schedule[total_step])
                total_step += 1

    def evaluate(
        self,
        eval_dataloader,
        compute_metrics,
        inputs_collator,
        **kwargs,
    ):
        """
        The method evaluate alignment.
        """

        all_metrics = []
        all_num_examples = []
        torch.cuda.empty_cache()
        with torch.no_grad():
            for inputs in tqdm(eval_dataloader, desc="Evaluating", leave=False):
                if inputs_collator is not None:
                    inputs = inputs_collator(inputs)
                b_s = inputs["input_ids"].shape[0]
                unit_location_dict = self._batch_process_unit_location(
                    inputs,
                )
                _, counterfactual_outputs = self(
                    {"input_ids": inputs["input_ids"]},
                    [{"input_ids": inputs["source_input_ids"]}],
                    unit_location_dict,
                    subspaces=inputs["subspaces"] if "subspaces" in inputs else None,
                )
                eval_metrics = compute_metrics(
                    [counterfactual_outputs.logits], [inputs["labels"]]
                )
                all_metrics += [eval_metrics]
                all_num_examples += [b_s]
        result = weighted_average(all_metrics, all_num_examples)

        return result
