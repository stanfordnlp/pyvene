import json, logging, torch, types
import numpy as np
from collections import OrderedDict
from collections.abc import Mapping
from typing import List, Optional, Tuple, Union, Dict, Any

from .constants import *
from .basic_utils import *
from .modeling_utils import *
from .intervention_utils import *
from .interventions import *
from .configuration_intervenable_model import (
    IntervenableConfig,
    RepresentationConfig,
)
from .interventions import (
    TrainableIntervention,
    SkipIntervention,
    CollectIntervention,
    BoundlessRotatedSpaceIntervention,
    InterventionOutput
)

from torch import optim
from transformers import get_linear_schedule_with_warmup
from dataclasses import dataclass
from transformers.utils import ModelOutput
from tqdm import tqdm, trange

try:
    import nnsight
except:
    print("nnsight is not detected. Please install via 'pip install nnsight' for nnsight backend.")


@dataclass
class IntervenableModelOutput(ModelOutput):
    """
    Output of the IntervenableModel, including original outputs, intervened outputs, and collected activations.
    """
    original_outputs: Optional[Any] = None
    intervened_outputs: Optional[Any] = None
    collected_activations: Optional[Any] = None


class BaseModel(nn.Module):
    """
    Base model class for sharing static vars and methods.
    """

    def __init__(self, config, model, backend=None, **kwargs):
        super().__init__()
        if isinstance(config, dict) or isinstance(config, list):
            config = IntervenableConfig(
                representations = config
            )
        self.config = config

        self.mode = config.mode
        intervention_type = config.intervention_types

        ###
        # nnsight wiring.
        #
        # Every intervention in pyvene is now executed through nnsight: the model
        # is run inside an `nnsight` trace and activations are read/written via
        # Envoy `.input` / `.output` accessors instead of raw
        # `register_forward_hook` calls. We accept either a plain `torch.nn.Module`
        # (which we wrap) or an already-wrapped nnsight model. `self._ns` is the
        # wrapper used for tracing; `self.model` stays the *raw* module so all
        # config / parameter / device / save logic is unchanged.
        ###
        if isinstance(model, nnsight.NNsight):
            self._ns = model
            model = unwrap_model(model)
        else:
            self._ns = nnsight.NNsight(model)

        self.is_model_stateless = is_stateless(model)
        self.config.model_type = str(type(model)) # backfill
        self.use_fast = kwargs["use_fast"] if "use_fast" in kwargs else False
        # if as_adaptor is turn on, we pass in the input args to the intervention
        self.as_adaptor = kwargs["as_adaptor"] if "as_adaptor" in kwargs else False
        if self.as_adaptor:
            logging.warning(
                "as_adaptor is turned on. This means the intervention will take "
                "the input arguments of the intervening module as well."
            )
        self.model_has_grad = False
        if self.use_fast:
            logging.warning(
                "Detected use_fast=True means the intervention location "
                "will be static within a batch.\n\nIn case multiple "
                "location tags are passed only the first one will "
                "be considered"
            )
        # each representation can get a different intervention type
        if type(intervention_type) == list:
            assert len(intervention_type) == len(
                config.representations
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
        self.representations = {}
        self.interventions = torch.nn.ModuleDict({})
        self.intervention_hooks = {}
        self._key_collision_counter = {}
        self.return_collect_activations = False
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
            config.representations
        ):
            _key = self._get_representation_key(representation)

            if representation.intervention is not None:
                intervention = representation.intervention
                intervention.use_fast = self.use_fast
            else:
                intervention_function = (
                    intervention_type
                    if type(intervention_type) != list
                    else intervention_type[i]
                )
                all_metadata = representation._asdict()
                component_dim = get_dimension_by_component(
                    get_internal_model_type(model), model.config, 
                    representation.component
                )
                if component_dim is not None:
                    component_dim *= int(representation.max_number_of_units)
                all_metadata["embed_dim"] = component_dim
                all_metadata["use_fast"] = self.use_fast
                intervention = intervention_function(
                    **all_metadata
                )
                
            if representation.intervention_link_key in self._intervention_pointers:
                self._intervention_reverse_link[
                    _key
                ] = f"link#{representation.intervention_link_key}"
                intervention = self._intervention_pointers[
                    representation.intervention_link_key
                ]
            elif representation.intervention_link_key is not None:
                self._intervention_pointers[
                    representation.intervention_link_key
                ] = intervention
                self._intervention_reverse_link[
                    _key
                ] = f"link#{representation.intervention_link_key}"
                    
            if isinstance(
                intervention,
                CollectIntervention
            ):
                self.return_collect_activations = True
            
            module_hook = get_module_hook(
                self._ns, representation
            )
            self.representations[_key] = representation
            if isinstance(intervention, types.FunctionType):
                self.interventions[_key] = LambdaIntervention(intervention)
            else:
                self.interventions[_key] = intervention
            self.intervention_hooks[_key] = module_hook
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
        if self.config.sorted_keys is not None:
            logging.warning(
                "The key is provided in the config. "
                "Assuming this is loaded from a pretrained module."
            )
        if (
            self.config.sorted_keys is not None
            or "intervenables_sort_fn" not in kwargs
        ):
            self.sorted_keys = _original_key_order
        else:
            # the key order is independent of group, it is used to read out intervention locations.
            self.sorted_keys = kwargs["intervenables_sort_fn"](
                model, self.representations
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
            for _key in self.sorted_keys:
                representation = self.representations[_key]
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
            for _key in self.sorted_keys:
                self._intervention_group[_group_key_inc] = [_key]
                _group_key_inc += 1
        # sort group key with ascending order
        self._intervention_group = OrderedDict(sorted(self._intervention_group.items()))

        # cached swap-in activations
        self.activations = {}
        # cached swapped activations (hot)
        self.hot_activations = {}

        self.full_intervention_outputs = []
        
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
        self.trainable_model_parameters = {}

    def __str__(self):
        """
        Print out basic info about this intervenable instance
        """
        attr_dict = {
            "model_type": self.model_type,
            "intervention_types": self.intervention_types,
            "alignabls": self.sorted_keys,
            "mode": self.mode,
        }
        return json.dumps(attr_dict, indent=4)

    def _get_representation_key(self, representation):
        """
        Provide unique key for each intervention
        """
        l = representation.layer
        c = representation.component
        u = representation.unit
        n = representation.max_number_of_units
        _u = u.replace(".", "_") # this will need internal functions to be changed as well.
        if "." in c:
            _c = c.replace(".", "_")
            # string access for sure
            key_proposal = f"comp_{_c}_unit_{_u}_nunit_{n}"
        else:
            key_proposal = f"layer_{l}_comp_{c}_unit_{_u}_nunit_{n}"
        if key_proposal not in self._key_collision_counter:
            self._key_collision_counter[key_proposal] = 0
        else:
            self._key_collision_counter[key_proposal] += 1
        return f"{key_proposal}#{self._key_collision_counter[key_proposal]}"
    
    def get_trainable_parameters(self):
        """
        Return trainable params as key value pairs
        """
        ret_params = []
        for k, v in self.interventions.items():
            if isinstance(v, TrainableIntervention):
                ret_params += [p for p in v.parameters()]
        for p in self.model.parameters():
            if p.requires_grad:
                ret_params += [p]
        return ret_params
    
    def named_parameters(self, recurse=True):
        """
        The above, but for HuggingFace.
        """
        ret_params = []
        for k, v in self.interventions.items():
            if isinstance(v, TrainableIntervention):
                ret_params += [(k + '.' + n, p) for n, p in v.named_parameters()]
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                ret_params += [('model.' + n, p)]
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
            if isinstance(v, BoundlessRotatedSpaceIntervention) or \
                isinstance(v, SigmoidMaskIntervention):
                v.set_temperature(temp)

    def enable_model_gradients(self):
        """
        Enable gradient in the model
        """
        # Unfreeze all model weights
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True 
        self.model_has_grad = True
                
    def disable_model_gradients(self):
        """
        Disable gradient in the model
        """
        # Freeze all model weights
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model_has_grad = False
            
    def disable_intervention_gradients(self):
        """
        Disable gradient in the trainable intervention
        """
        # Freeze all intervention weights
        pass

    def set_device(self, device, set_model=True):
        """
        Set device of interventions and the model
        """
        for k, v in self.interventions.items():
            v.to(device)
        if set_model:
            self.model.to(device)

    def get_device(self):
        """
        Get device of interventions and the model
        """
        return self.model.device

    def count_parameters(self, include_model=False):
        """
        Set device of interventions and the model
        """
        _linked_key_set = set([])
        total_parameters = 0
        for k, v in self.interventions.items():
            if isinstance(v, TrainableIntervention):
                if k in self._intervention_reverse_link:
                    if not self._intervention_reverse_link[k] in _linked_key_set:
                        _linked_key_set.add(self._intervention_reverse_link[k])
                        total_parameters += count_parameters(v)
                else:
                    total_parameters += count_parameters(v)
        if include_model:
            total_parameters += sum(
                p.numel() for p in self.model.parameters() if p.requires_grad)
        return total_parameters

    def set_zero_grad(self):
        """
        Set device of interventions and the model
        """
        for k, v in self.interventions.items():
            if isinstance(v, TrainableIntervention):
                v.zero_grad()

    def zero_grad(self):
        """
        The above, but for HuggingFace.
        """
        for k, v in self.interventions.items():
            if isinstance(v, TrainableIntervention):
                v.zero_grad()

    def _input_validation(
        self,
        base,
        sources,
        unit_locations,
        activations_sources,
        subspaces,
    ):
        """Fail fast input validation"""
        if self.mode == "parallel" and unit_locations is not None:
            assert "sources->base" in unit_locations or "base" in unit_locations
        elif activations_sources is None and unit_locations is not None and self.mode == "serial":
            assert "sources->base" not in unit_locations
        
        # sources may contain None, but length should match
        if sources is not None and not (len(sources) == 1 and sources[0] == None):
            if len(sources) != len(self._intervention_group):
                raise ValueError(
                    f"Source length {len(sources)} is not "
                    f"equal to intervention length {len(self._intervention_group)}."
                )
        elif activations_sources is not None:
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

    def _gather_intervention_output(
        self, output, representations_key, unit_locations
    ) -> torch.Tensor:
        """
        Gather intervening activations from the output based on indices
        """

        if (
            representations_key in self._intervention_reverse_link
            and self._intervention_reverse_link[representations_key]
            in self.hot_activations
        ):
            # hot gather
            # clone is needed here by acting as a different module
            # to avoid gradient conflict.
            #
            # enable the following line when an error is hit
            # torch.autograd.set_detect_anomaly(True)
            selected_output = self.hot_activations[
                self._intervention_reverse_link[representations_key]
            ]
        else:
            # data structure casting
            if isinstance(output, tuple):
                original_output = output[0].clone()
            elif isinstance(output, dict):
                original_output = output[list(output.keys())[0]].clone()
            else:
                original_output = output.clone()
            # for non-sequence models, there is no concept of
            # unit location anyway.
            if unit_locations is None:
                return original_output
            # gather subcomponent
            original_output = output_to_subcomponent(
                original_output,
                self.representations[
                    representations_key
                ].component,
                self.model_type,
                self.model_config,
            )

            # gather based on intervention locations
            selected_output = gather_neurons(
                original_output,
                self.representations[
                    representations_key
                ].unit,
                unit_locations,
                device=self.get_device()
            )

        return selected_output

    def _scatter_intervention_output(
        self,
        output,
        intervened_representation,
        representations_key,
        unit_locations,
    ) -> torch.Tensor:
        """
        Scatter in the intervened activations in the output
        """
        # data structure casting
        if isinstance(output, tuple):
            original_output = output[0]
        elif isinstance(output, dict):
            original_output = output[list(output.keys())[0]]
        else:
            original_output = output
        # for non-sequence-based models, we simply replace
        # all the activations.
        if unit_locations is None:
            original_output[:] = intervened_representation[:]
            return original_output
        
        component = self.representations[
            representations_key
        ].component
        unit = self.representations[
            representations_key
        ].unit
        
        # scatter in-place
        _ = scatter_neurons(
            original_output,
            intervened_representation,
            component,
            unit,
            unit_locations,
            self.model_type,
            self.model_config,
            self.use_fast,
            device=self.get_device()
        )
        
        return original_output
    
    def _broadcast_unit_locations(
        self,
        batch_size,
        unit_locations
    ):
        if unit_locations is None:
            # this means, we don't filter based on location at all.
            return {"sources->base": ([None]*len(self.interventions), [None]*len(self.interventions))}
        
        if self.mode == "parallel":
            _unit_locations = {}
            for k, v in unit_locations.items():
                # special broadcast for base-only interventions
                is_base_only = False
                if k == "base":
                    is_base_only = True
                    k = "sources->base"
                if isinstance(v, int):
                    if is_base_only:
                        _unit_locations[k] = (None, [[[v]]*batch_size]*len(self.interventions))
                    else:
                        _unit_locations[k] = (
                            [[[v]]*batch_size]*len(self.interventions), 
                            [[[v]]*batch_size]*len(self.interventions)
                        )
                    self.use_fast = True
                elif len(v) == 2 and isinstance(v[0], int) and isinstance(v[1], int):
                    _unit_locations[k] = (
                        [[[v[0]]]*batch_size]*len(self.interventions), 
                        [[[v[1]]]*batch_size]*len(self.interventions)
                    )
                    self.use_fast = True
                elif len(v) == 2 and v[0] == None and isinstance(v[1], int):
                    _unit_locations[k] = (None, [[[v[1]]]*batch_size]*len(self.interventions))
                    self.use_fast = True
                elif len(v) == 2 and isinstance(v[0], int) and v[1] == None:
                    _unit_locations[k] = ([[[v[0]]]*batch_size]*len(self.interventions), None)
                    self.use_fast = True
                elif isinstance(v, list) and get_list_depth(v) == 1:
                    # [0,1,2,3] -> [[[0,1,2,3]]], ...
                    if is_base_only:
                        _unit_locations[k] = (None, [[v]*batch_size]*len(self.interventions))
                    else:
                        _unit_locations[k] = (
                            [[v]*batch_size]*len(self.interventions), 
                            [[v]*batch_size]*len(self.interventions)
                        )
                    self.use_fast = True
                else:
                    if is_base_only:
                        _unit_locations[k] = (None, v)
                    else:
                        _unit_locations[k] = v
        elif self.mode == "serial":
            _unit_locations = {}
            for k, v in unit_locations.items():
                if isinstance(v, int):
                    _unit_locations[k] = (
                        [[[v]]*batch_size]*len(self.interventions), 
                        [[[v]]*batch_size]*len(self.interventions)
                    )
                    self.use_fast = True
                elif len(v) == 2 and isinstance(v[0], int) and isinstance(v[1], int):
                    _unit_locations[k] = (
                        [[[v[0]]]*batch_size]*len(self.interventions), 
                        [[[v[1]]]*batch_size]*len(self.interventions)
                    )
                    self.use_fast = True
                elif len(v) == 2 and v[0] == None and isinstance(v[1], int):
                    _unit_locations[k] = (None, [[[v[1]]]*batch_size]*len(self.interventions))
                    self.use_fast = True
                elif len(v) == 2 and isinstance(v[0], int) and v[1] == None:
                    _unit_locations[k] = ([[[v[0]]]*batch_size]*len(self.interventions), None)
                    self.use_fast = True
                elif isinstance(v, list) and get_list_depth(v) == 1:
                    # [0,1,2,3] -> [[[0,1,2,3]]], ...
                    _unit_locations[k] = (
                        [[v]*batch_size]*len(self.interventions), 
                        [[v]*batch_size]*len(self.interventions)
                    )
                    self.use_fast = True
                else:
                    _unit_locations[k] = v
        else:
            raise ValueError(f"The mode {self.mode} is not supported.")
        return _unit_locations
    
    def _broadcast_source_representations(
        self,
        source_representations
    ):
        """Broadcast simple inputs to a dict"""
        _source_representations = {}
        if isinstance(source_representations, dict) or source_representations is None:
            # pass to broadcast for advance usage
            _source_representations = source_representations
        elif isinstance(source_representations, list):
            for i, key in enumerate(self.sorted_keys):
                _source_representations[key] = source_representations[i]
        elif isinstance(source_representations, torch.Tensor):
            for key in self.sorted_keys:
                _source_representations[key] = source_representations
        else:
            raise ValueError(
                "Accept input type for source_representations is [Dict, List, torch.Tensor]"
            )
        return _source_representations
            
    def _broadcast_sources(
        self,
        sources
    ):
        """Broadcast simple inputs to a dict"""
        _sources = sources
        if len(sources) == 1 and len(self._intervention_group) > 1:
            for _ in range(len(self._intervention_group)-1):
                _sources += [sources[0]]
        else:
            _sources = sources
        return _sources
    
    def _broadcast_subspaces(
        self,
        batch_size,
        subspaces
    ):
        """Broadcast simple subspaces input"""
        _subspaces = subspaces
        if isinstance(subspaces, int):
            _subspaces = [[[subspaces]]*batch_size]*len(self.interventions)
            
        elif isinstance(subspaces, list) and isinstance(subspaces[0], int):
            _subspaces = [[subspaces]*batch_size]*len(self.interventions)
        else:
            # TODO: subspaces is easier to add more broadcast majic.
            pass
        return _subspaces
    
    def forward(self, **kwargs):
        raise NotImplementedError("Please Implement this method")

    def generate(self, **kwargs):
        raise NotImplementedError("Please Implement this method")
        

class IntervenableModel(BaseModel):
    """
    Intervenable model powered by nnsight.

    The model is executed inside an ``nnsight`` trace and activations are read /
    written declaratively through Envoy ``.input`` / ``.output`` accessors — no
    raw ``register_forward_hook`` calls are used. Source activations are gathered
    in one trace and applied to the base in another; gradients flow back through
    trainable interventions exactly as with a normal forward pass.
    """
    BACKEND = "nnsight"

    def __init__(self, config, model, **kwargs):
        super().__init__(config, model, "nnsight", **kwargs)

    def _reset_hook_count(self):
        """
        Reset the per-call intervention state counters.
        """
        self._key_getter_call_counter = dict.fromkeys(self._key_getter_call_counter, 0)
        self._key_setter_call_counter = dict.fromkeys(self._key_setter_call_counter, 0)
        for k, _ in self._intervention_state.items():
            self._intervention_state[k].reset()

    def _cleanup_states(self, skip_activation_gc=False):
        """
        Clean up all old in-memory states of interventions.
        """
        self._is_generation = False
        self._reset_hook_count()
        if not skip_activation_gc:
            self.activations.clear()
            self.hot_activations.clear()
            self._batched_setter_activation_select.clear()
        else:
            self.activations = {}
            self.hot_activations = {}
            self._batched_setter_activation_select = {}

    def save(
        self, save_directory, save_to_hf_hub=False, hf_repo_name="my-awesome-model",
        include_model=False
    ):
        """
        Save interventions to disk or hub
        """
        if save_to_hf_hub:
            from huggingface_hub import HfApi

            api = HfApi()

        create_directory(save_directory)

        saving_config = copy.deepcopy(self.config)
        saving_config.sorted_keys = self.sorted_keys
        saving_config.model_type = str(
            saving_config.model_type
        )
        saving_config.intervention_types = []
        saving_config.intervention_dimensions = []
        saving_config.intervention_constant_sources = []
        
        # handle constant source reprs if passed in.
        serialized_representations = []
        for reprs in saving_config.representations:
            serialized_reprs = {}
            for k, v in reprs._asdict().items():
                if k == "hidden_source_representation":
                    continue
                if k == "source_representation":
                    # hidden flag only set here
                    if v is not None:
                        serialized_reprs["hidden_source_representation"] = True
                    serialized_reprs[k] = None
                elif k == "intervention_type":
                    serialized_reprs[k] = None
                elif k == "intervention":
                    serialized_reprs[k] = None
                else:
                    serialized_reprs[k] = v
            serialized_representations += [
                RepresentationConfig(**serialized_reprs)
            ]
        saving_config.representations = \
            serialized_representations
        
        for k, v in self.interventions.items():
            intervention = v
            saving_config.intervention_types += [str(type(intervention))]
            binary_filename = f"intkey_{k}.bin"
            # save intervention binary file
            if isinstance(intervention, TrainableIntervention) or \
                intervention.source_representation is not None:
                # logging.info(f"Saving trainable intervention to {binary_filename}.")
                torch.save(
                    intervention.state_dict(),
                    os.path.join(save_directory, binary_filename),
                )
                if save_to_hf_hub:
                    # push to huggingface hub
                    try:
                        api.create_repo(hf_repo_name)
                    except:
                        logging.info(
                            f"Uploading: {binary_filename}, but skipping creating the repo since "
                            f"either {hf_repo_name} exists or having authentication error."
                        )
                    api.upload_file(
                        path_or_fileobj=os.path.join(save_directory, binary_filename),
                        path_in_repo=binary_filename,
                        repo_id=hf_repo_name,
                        repo_type="model",
                    )
            if intervention.interchange_dim is None:
                saving_config.intervention_dimensions += [None]
            else:
                saving_config.intervention_dimensions += [intervention.interchange_dim.tolist()]
            saving_config.intervention_constant_sources += [intervention.is_source_constant]

        # save model's trainable parameters as well
        if include_model:
            model_state_dict = {}
            model_binary_filename = "pytorch_model.bin"
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    model_state_dict[n] = p
            torch.save(model_state_dict, os.path.join(save_directory, model_binary_filename))
            
        # save metadata config
        saving_config.save_pretrained(save_directory)
        if save_to_hf_hub:
            # push to huggingface hub
            try:
                api.create_repo(hf_repo_name)
            except:
                logging.info(
                    f"Uploading the config, Skipping creating the repo since "
                    f"either {hf_repo_name} exists or having authentication error."
                )
            api.upload_file(
                path_or_fileobj=os.path.join(save_directory, "config.json"),
                path_in_repo="config.json",
                repo_id=hf_repo_name,
                repo_type="model",
            )

    @staticmethod
    def load(
        load_directory, model, local_directory=None, from_huggingface_hub=False,
        include_model=False
    ):
        """
        Load interventions from disk or hub
        """
        if not os.path.exists(load_directory) or from_huggingface_hub:
            from_huggingface_hub = True
            
            from huggingface_hub import snapshot_download
            load_directory = snapshot_download(
                repo_id=load_directory,
                local_dir=local_directory,
            )

        # load config
        saving_config = IntervenableConfig.from_pretrained(load_directory)
        casted_intervention_types = []

        for type_str in saving_config.intervention_types:
            casted_intervention_types += [get_type_from_string(type_str)]
        saving_config.intervention_types = (
            casted_intervention_types
        )
        casted_representations = []
        for (
            representation_opts
        ) in saving_config.representations:
            casted_representations += [
                RepresentationConfig(*representation_opts)
            ]
        saving_config.representations = casted_representations
        intervenable = IntervenableModel(saving_config, model)

        # load binary files
        for i, (k, v) in enumerate(intervenable.interventions.items()):
            intervention = v
            binary_filename = f"intkey_{k}.bin"
            intervention.is_source_constant = \
                saving_config.intervention_constant_sources[i]
            dim = saving_config.intervention_dimensions[i]
            if dim is None:
                # Infer interchange dimension from component name to be compatible with old versions
                component_name = saving_config.representations[i].component
                if component_name.startswith("head_"):
                    dim = model.config.hidden_size // model.config.num_attention_heads
                else:
                    dim = model.config.hidden_size

            intervention.set_interchange_dim(dim)
            if saving_config.intervention_constant_sources[i] and \
                not isinstance(intervention, ZeroIntervention) and \
                not isinstance(intervention, SourcelessIntervention):
                # logging.warn(f"Loading trainable intervention from {binary_filename}.")
                saved_state_dict = torch.load(os.path.join(load_directory, binary_filename))
                try:
                    intervention.register_buffer(
                        'source_representation', saved_state_dict['source_representation']
                    )
                except:
                    intervention.source_representation = saved_state_dict['source_representation']
            elif isinstance(intervention, TrainableIntervention):
                saved_state_dict = torch.load(os.path.join(load_directory, binary_filename))
                intervention.load_state_dict(saved_state_dict)

        # load model's trainable parameters as well
        if include_model:
            model_binary_filename = "pytorch_model.bin"
            saved_model_state_dict = torch.load(os.path.join(load_directory, model_binary_filename))
            intervenable.model.load_state_dict(saved_model_state_dict, strict=False)

        return intervenable

    def save_intervention(self, save_directory, include_model=True):
        """
        Instead of saving the metadata with artifacts, it only saves artifacts such as
        trainable weights. This is not a static method, and returns nothing.
        """
        create_directory(save_directory)
        
        # save binary files
        for k, v in self.interventions.items():
            intervention = v
            binary_filename = f"intkey_{k}.bin"
            # save intervention binary file
            if isinstance(intervention, TrainableIntervention):
                torch.save(intervention.state_dict(),
                    os.path.join(save_directory, binary_filename))

        # save model's trainable parameters as well
        if include_model:
            model_state_dict = {}
            model_binary_filename = "pytorch_model.bin"
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    model_state_dict[n] = p
            torch.save(model_state_dict, os.path.join(save_directory, model_binary_filename))
    
    def load_intervention(self, load_directory, include_model=True):
        """
        Instead of creating an new object, this function loads existing weights onto
        the current object. This is not a static method, and returns nothing.
        """
        # load binary files
        for i, (k, v) in enumerate(self.interventions.items()):
            intervention = v
            binary_filename = f"intkey_{k}.bin"
            if isinstance(intervention, TrainableIntervention):
                saved_state_dict = torch.load(
                    os.path.join(load_directory, binary_filename), map_location='cuda:0')
                intervention.load_state_dict(saved_state_dict)

        # load model's trainable parameters as well
        if include_model:
            model_binary_filename = "pytorch_model.bin"
            saved_model_state_dict = torch.load(os.path.join(load_directory, model_binary_filename))
            self.model.load_state_dict(saved_model_state_dict, strict=False)


    def _trace(self, inputs, **model_kwargs):
        """Open an nnsight trace over ``inputs``.

        For a plain ``NNsight`` wrapper (the common case — pyvene wraps the raw
        HF/MLP/GRU module) a dict input is unpacked as keyword args, mirroring the
        old ``self.model(**base)`` call. An ``nnsight.LanguageModel`` consumes the
        input directly (it handles tokenization / dict packing itself).
        """
        if isinstance(self._ns, nnsight.LanguageModel):
            return self._ns.trace(inputs, **model_kwargs)
        if isinstance(inputs, Mapping):
            return self._ns.trace(**inputs, **model_kwargs)
        return self._ns.trace(inputs, **model_kwargs)

    def _run_model(self, base, **model_kwargs):
        """Run an un-intervened forward and return the model output."""
        with self._trace(base, **model_kwargs):
            output = self._ns.output.save()
        return output

    def _read_module_activation(self, module_hook, hook_type):
        """Return the live tensor (or tuple/dict) at an Envoy's input/output."""
        envoy = module_hook[0]
        if hook_type == CONST_INPUT_HOOK:
            return envoy.input
        return envoy.output

    def _reconcile_stateful_cached_activations(
        self, key, intervening_activations, intervening_unit_locations
    ):
        """Return the cached source activation for ``key``.

        For stateless models (transformers/MLPs) this is simply the saved source
        activation. Stateful reconciliation (GRU) is only reachable for non
        source-constant interventions, which the engine does not exercise.
        """
        if key not in self.activations:
            return None
        return self.activations[key]

    def _capture_activation(self, key, unit_locations):
        """Getter: gather the aligned activation at ``key`` and cache it.

        Must run inside an nnsight trace over the source input.
        """
        intervention = self.interventions[key]
        module_hook = self.intervention_hooks[key]
        hook_type = module_hook[1]

        if isinstance(intervention, SkipIntervention):
            # a skip reads the *input* to the module
            value = self._read_module_activation(module_hook, CONST_INPUT_HOOK)
        else:
            value = self._read_module_activation(module_hook, hook_type)

        selected_output = self._gather_intervention_output(
            value, key, unit_locations
        )
        # persist past the trace so it can be swapped into the base run
        self.activations[key] = selected_output.save()
        self._intervention_state[key].inc_getter_version()

    def _apply_intervention(
        self, key, unit_locations_base, subspace, intervention_additional_kwargs
    ):
        """Setter: read, transform and write back the activation at ``key``.

        Must run inside an nnsight trace over the (intervened) input.
        """
        intervention = self.interventions[key]
        module_hook = self.intervention_hooks[key]
        hook_type = module_hook[1]
        if intervention_additional_kwargs is None:
            intervention_additional_kwargs = {}

        value = self._read_module_activation(module_hook, hook_type)

        selected_output = self._gather_intervention_output(
            value, key, unit_locations_base
        )
        if not self.is_model_stateless:
            selected_output = selected_output.clone()

        if self.as_adaptor:
            adaptor_input = self._read_module_activation(
                module_hook, CONST_INPUT_HOOK
            )
            selected_input = self._gather_intervention_output(
                adaptor_input, key, unit_locations_base
            )
            intervention_additional_kwargs = dict(intervention_additional_kwargs)
            intervention_additional_kwargs["args"] = selected_input

        if isinstance(intervention, CollectIntervention):
            intervened_representation = do_intervention(
                selected_output,
                None,
                intervention,
                subspace,
                **intervention_additional_kwargs,
            )
            # support collection during generation by accumulating a list
            if key not in self.activations:
                self.activations[key] = [intervened_representation.save()]
            else:
                self.activations[key].append(intervened_representation.save())
            # no-op to the output
            return

        if not isinstance(intervention, LambdaIntervention) and \
                intervention.is_source_constant:
            raw_intervened_representation = do_intervention(
                selected_output,
                None,
                intervention,
                subspace,
                **intervention_additional_kwargs,
            )
            if isinstance(raw_intervened_representation, InterventionOutput):
                self.full_intervention_outputs.append(raw_intervened_representation)
                intervened_representation = raw_intervened_representation.output
            else:
                intervened_representation = raw_intervened_representation
        else:
            intervened_representation = do_intervention(
                selected_output,
                self._reconcile_stateful_cached_activations(
                    key, selected_output, unit_locations_base
                ),
                intervention,
                subspace,
                **intervention_additional_kwargs,
            )

        if intervened_representation is None:
            return

        # linked interventions share their swapped ("hot") activation
        if key in self._intervention_reverse_link:
            self.hot_activations[
                self._intervention_reverse_link[key]
            ] = intervened_representation.clone()

        self._scatter_intervention_output(
            value, intervened_representation, key, unit_locations_base
        )
        self._intervention_state[key].inc_setter_version()

    def _subspace_for(self, key, subspaces):
        """Pick the subspace entry aligned with ``key`` (or ``None``)."""
        if subspaces is None:
            return None
        return subspaces[self.sorted_keys.index(key)]

    def _should_intervene(self, key):
        """Whether ``key`` has an activation/source ready to be applied."""
        return (
            key in self.activations
            or isinstance(self.interventions[key], LambdaIntervention)
            or self.interventions[key].is_source_constant
        )

    def _collect_parallel_sources(
        self, sources, unit_locations_sources, activations_sources
    ):
        """Cache every source activation (parallel mode)."""
        if activations_sources is None:
            assert len(sources) == len(self._intervention_group)
            for group_id, keys in self._intervention_group.items():
                if sources[group_id] is None:
                    continue  # smart jump for advance usage only
                with self._trace(sources[group_id]):
                    for key in keys:
                        self._capture_activation(
                            key,
                            unit_locations_sources[self.sorted_keys.index(key)],
                        )
        else:
            self.activations = activations_sources
            for passed_in_key in self.activations:
                assert passed_in_key in self.sorted_keys

    def _intervene_parallel(
        self, base, sources, unit_locations, activations_sources,
        subspaces, intervention_additional_kwargs, model_kwargs,
    ):
        """Parallel mode: collect all sources, then swap them into base at once."""
        unit_locations_sources = unit_locations["sources->base"][0]
        unit_locations_base = unit_locations["sources->base"][1]

        self._collect_parallel_sources(
            sources, unit_locations_sources, activations_sources
        )

        with self._trace(base, **model_kwargs):
            for group_id, keys in self._intervention_group.items():
                for key in keys:
                    if self._should_intervene(key):
                        self._apply_intervention(
                            key,
                            unit_locations_base[self.sorted_keys.index(key)],
                            self._subspace_for(key, subspaces),
                            intervention_additional_kwargs,
                        )
            counterfactual_outputs = self._ns.output.save()
        return counterfactual_outputs

    def _intervene_serial(
        self, base, sources, unit_locations, activations_sources,
        subspaces, intervention_additional_kwargs, model_kwargs,
    ):
        """Serial mode: thread interventions source_0 -> source_1 -> ... -> base.

        Each stage runs in a single trace where the *previous* group's
        interventions are re-applied (so the current source sees them) while the
        current group's activation is captured. The final stage runs the base.
        """
        group_ids = list(self._intervention_group.keys())
        n_groups = len(group_ids)

        # interventions that must be re-applied on the *next* trace, as
        # (key, base_unit_locations, subspace) tuples.
        pending = []

        for idx, group_id in enumerate(group_ids):
            keys = self._intervention_group[group_id]
            if idx != n_groups - 1:
                loc_key = f"source_{group_id}->source_{group_id+1}"
            else:
                loc_key = f"source_{group_id}->base"

            stage_input = sources[group_id]
            if stage_input is None:
                continue  # smart jump for advance usage only

            unit_locations_source = unit_locations[loc_key][0]
            unit_locations_base = unit_locations[loc_key][1]

            with self._trace(stage_input):
                # re-apply already-resolved interventions so this source sees them
                for (p_key, p_loc, p_sub) in pending:
                    self._apply_intervention(
                        p_key, p_loc, p_sub, intervention_additional_kwargs
                    )
                # capture this group's activation from the (intervened) source
                if activations_sources is None:
                    for key_id, key in enumerate(keys):
                        if unit_locations_source[key_id] is None:
                            continue
                        self._capture_activation(key, unit_locations_source[key_id])
                else:
                    for key in keys:
                        self.activations[key] = activations_sources[key]

            # this group's interventions get applied on the next trace
            pending = []
            for key_id, key in enumerate(keys):
                if self._should_intervene(key):
                    pending.append(
                        (key, unit_locations_base[key_id], self._subspace_for(key, subspaces))
                    )

        with self._trace(base, **model_kwargs):
            for (p_key, p_loc, p_sub) in pending:
                self._apply_intervention(
                    p_key, p_loc, p_sub, intervention_additional_kwargs
                )
            counterfactual_outputs = self._ns.output.save()
        return counterfactual_outputs

    def _output_validation(self):
        """Safe guard: stateless interventions fire their getter/setter once."""
        if self.is_model_stateless:
            for k, v in self._intervention_state.items():
                if v.getter_version() > 1 or v.setter_version() > 1:
                    raise Exception(
                        f"For stateless model, each getter and setter "
                        f"should be called only once: {self._intervention_state}"
                    )

    def forward(
        self,
        base,
        sources: Optional[List] = None,
        unit_locations: Optional[Dict] = None,
        source_representations: Optional[Dict] = None,
        subspaces: Optional[List] = None,
        labels: Optional[torch.LongTensor] = None,
        output_original_output: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        intervention_additional_kwargs: Optional[Dict] = None,
    ):
        """
        Main forward function that serves a wrapper to
        actual model forward calls. It will use forward
        hooks to do interventions.

        In essence, sources will lead to getter hooks to
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

        An intervention could be initialized with subspace partition as,
        [[... subspace_1 ...], [... subspace_2 ...], [rest]]

        An intervention may be targeting a specific partition.

        This input field should look like something like,
        [
            [[subspace indices], [subspace indices]], <- for the first intervention
            None,                                     <- for the second intervention
            [[subspace indices], [subspace indices]]
        ]

        Only setter (where do_intervention is called) needs this field.

        *We assume base and source targeting the same subspace for now.
        *We assume only a single space is targeted for now (although 2d list is provided).

        Since we now support group-based intervention, the number of sources
        should be equal to the total number of groups.
        """
        activations_sources = source_representations
        if sources is not None and not isinstance(sources, list):
            sources = [sources]

        self.full_intervention_outputs.clear()

        self._cleanup_states()

        # if no source input or intervention, we return base
        if sources is None and activations_sources is None \
            and unit_locations is None and len(self.interventions) == 0:
            return self._run_model(base), None
        # broadcast
        unit_locations = self._broadcast_unit_locations(get_batch_size(base), unit_locations)
        sources = [None]*len(self._intervention_group) if sources is None else sources
        sources = self._broadcast_sources(sources)
        activations_sources = self._broadcast_source_representations(activations_sources)
        subspaces = self._broadcast_subspaces(get_batch_size(base), subspaces)

        self._input_validation(
            base,
            sources,
            unit_locations,
            activations_sources,
            subspaces,
        )

        # extra model kwargs threaded into the (intervened) forward
        model_kwargs = {}
        if labels is not None: # for training
            model_kwargs["labels"] = labels
        if use_cache is not None and 'use_cache' in self.model.config.to_dict(): # for transformer models
            model_kwargs["use_cache"] = use_cache

        base_outputs = None
        if output_original_output:
            # returning un-intervened output with gradients
            base_outputs = self._run_model(base)

        try:
            # intervene: collect source activations and apply them to base,
            # all through nnsight traces.
            if self.mode == "parallel":
                counterfactual_outputs = self._intervene_parallel(
                    base,
                    sources,
                    unit_locations,
                    activations_sources,
                    subspaces,
                    intervention_additional_kwargs,
                    model_kwargs,
                )
            elif self.mode == "serial":
                counterfactual_outputs = self._intervene_serial(
                    base,
                    sources,
                    unit_locations,
                    activations_sources,
                    subspaces,
                    intervention_additional_kwargs,
                    model_kwargs,
                )

            self._output_validation()

            collected_activations = []
            if self.return_collect_activations:
                for key in self.sorted_keys:
                    if isinstance(
                        self.interventions[key],
                        CollectIntervention
                    ):
                        collected_activations += self.activations[key]

        except Exception as e:
            raise e
        finally:
            self._cleanup_states(
                skip_activation_gc = \
                    (sources is None and activations_sources is not None) or \
                    self.return_collect_activations
            )
        
        if self.return_collect_activations:
            if return_dict:
                return IntervenableModelOutput(
                    original_outputs=base_outputs,
                    intervened_outputs=counterfactual_outputs,
                    collected_activations=collected_activations
                )
            
            return (base_outputs, collected_activations), counterfactual_outputs
        
        if return_dict:
            return IntervenableModelOutput(
                original_outputs=base_outputs,
                intervened_outputs=counterfactual_outputs,
                collected_activations=None
            )

        return base_outputs, counterfactual_outputs

    def generate(
        self,
        base,
        sources: Optional[List] = None,
        unit_locations: Optional[Dict] = None,
        source_representations: Optional[Dict] = None,
        intervene_on_prompt: bool = False,
        subspaces: Optional[List] = None,
        output_original_output: Optional[bool] = False,
        intervention_additional_kwargs: Optional[Dict] = None,
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
        activations_sources = source_representations
        if sources is not None and not isinstance(sources, list):
            sources = [sources]

        self.full_intervention_outputs.clear()
        self._cleanup_states()

        self._intervene_on_prompt = intervene_on_prompt
        self._is_generation = True

        if not intervene_on_prompt and unit_locations is None:
            # that means, we intervene on every generated tokens!
            unit_locations = {"base": 0}

        # broadcast
        unit_locations = self._broadcast_unit_locations(get_batch_size(base), unit_locations)
        sources = [None]*len(self._intervention_group) if sources is None else sources
        sources = self._broadcast_sources(sources)
        activations_sources = self._broadcast_source_representations(activations_sources)
        subspaces = self._broadcast_subspaces(get_batch_size(base), subspaces)

        self._input_validation(
            base,
            sources,
            unit_locations,
            activations_sources,
            subspaces,
        )

        if self.mode != "parallel":
            raise NotImplementedError(
                "Only parallel-mode generation is supported."
            )

        unit_locations_sources = unit_locations["sources->base"][0]
        unit_locations_base = unit_locations["sources->base"][1]

        base_outputs = None
        if output_original_output:
            # returning un-intervened generation
            base_outputs = self._generate_clean(base, kwargs)

        try:
            # collect source activations (a single forward per source group)
            self._collect_parallel_sources(
                sources, unit_locations_sources, activations_sources
            )

            counterfactual_outputs = self._generate_with_interventions(
                base,
                unit_locations_base,
                subspaces,
                intervention_additional_kwargs,
                kwargs,
            )

            collected_activations = []
            if self.return_collect_activations:
                for key in self.sorted_keys:
                    if isinstance(
                        self.interventions[key],
                        CollectIntervention
                    ):
                        collected_activations += self.activations[key]
        except Exception as e:
            raise e
        finally:
            self._is_generation = False
            self._cleanup_states(
                skip_activation_gc = \
                    (sources is None and activations_sources is not None) or \
                    self.return_collect_activations
            )

        if self.return_collect_activations:
            return (base_outputs, collected_activations), counterfactual_outputs

        return base_outputs, counterfactual_outputs

    def _generate_clean(self, base, gen_kwargs):
        """Un-intervened generation; returns the generated id tensor.

        No interventions are needed here, so we call ``generate`` eagerly (not as
        a ``with`` tracing context) — nnsight runs it and returns the output
        tensor directly.
        """
        if isinstance(self._ns, nnsight.LanguageModel):
            return self._ns.generate(base, **gen_kwargs)
        if isinstance(base, Mapping):
            return self._ns.generate(**base, **gen_kwargs)
        return self._ns.generate(base, **gen_kwargs)

    def _infer_generation_steps(self, base, gen_kwargs):
        """Best-effort count of generation steps for a bounded iteration.

        nnsight applies interventions per generated token inside a *bounded*
        ``tracer.iter[:n]`` loop; trailing access to ``tracer.result`` then works.
        We derive ``n`` from the generation kwargs. Returns ``None`` if unknown
        (callers fall back to an unbounded ``tracer.all()``).
        """
        if "max_new_tokens" in gen_kwargs:
            return int(gen_kwargs["max_new_tokens"])
        if "max_length" in gen_kwargs:
            prompt_len = None
            if isinstance(base, Mapping):
                for k in ("input_ids", "inputs_embeds"):
                    if k in base and hasattr(base[k], "shape"):
                        prompt_len = base[k].shape[1]
                        break
            if prompt_len is not None:
                return max(1, int(gen_kwargs["max_length"]) - prompt_len)
        return None

    def _apply_generation_steps(
        self, tracer, n_steps, unit_locations_base, subspaces,
        intervention_additional_kwargs,
    ):
        """Apply setters on each generated token and return the generated ids.

        Interventions run under ``no_grad`` — generation is inference, and the
        HF generate loop runs in ``no_grad`` mode, so a grad-tracked in-place
        write into a no-grad view would otherwise raise.
        """
        iterator = tracer.iter[0:n_steps] if n_steps is not None else tracer.all()
        for _ in iterator:
            with torch.no_grad():
                for group_id, keys in self._intervention_group.items():
                    for key in keys:
                        if self._should_intervene(key):
                            self._apply_intervention(
                                key,
                                unit_locations_base[self.sorted_keys.index(key)],
                                self._subspace_for(key, subspaces),
                                intervention_additional_kwargs,
                            )
        return tracer.result.save()

    def _generate_with_interventions(
        self, base, unit_locations_base, subspaces,
        intervention_additional_kwargs, gen_kwargs,
    ):
        """Run intervened generation, applying setters on each generated token."""
        n_steps = self._infer_generation_steps(base, gen_kwargs)
        # bound the per-token loop so trailing ``tracer.result`` access is valid;
        # forcing the exact new-token count keeps the bound from overshooting.
        gen_kwargs = dict(gen_kwargs)
        if n_steps is not None and "min_new_tokens" not in gen_kwargs:
            gen_kwargs["min_new_tokens"] = n_steps

        # the `with model.generate(...)` must be literal (see `_generate_clean`).
        if isinstance(self._ns, nnsight.LanguageModel):
            with self._ns.generate(base, **gen_kwargs) as tracer:
                out = self._apply_generation_steps(
                    tracer, n_steps, unit_locations_base, subspaces,
                    intervention_additional_kwargs,
                )
        elif isinstance(base, Mapping):
            with self._ns.generate(**base, **gen_kwargs) as tracer:
                out = self._apply_generation_steps(
                    tracer, n_steps, unit_locations_base, subspaces,
                    intervention_additional_kwargs,
                )
        else:
            with self._ns.generate(base, **gen_kwargs) as tracer:
                out = self._apply_generation_steps(
                    tracer, n_steps, unit_locations_base, subspaces,
                    intervention_additional_kwargs,
                )
        return out

    def _batch_process_unit_location(self, inputs):
        """
        Convert original data batch according
        to the intervenable settings.

        The function respects inputs in the following
        data format.


        Each location list in the raw input as,

        [[i, j, ...], [m, n, ...], ...] batched
        where i, j are the unit index, the outer
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
            for _, rep in self.representations.items():
                _curr_source_ind_inc = _curr_source_ind + 1
                _prefix = f"source_{_curr_source_ind}->base"
                _prefix_left = f"{_prefix}.0"
                _prefix_right = f"{_prefix}.1"
                _sub_loc_aggr_left = []  # 3d
                _sub_loc_aggr_right = []  # 3d
                for sub_loc in rep.unit.split("."):
                    _sub_loc_aggr_left += [inputs[f"{_prefix_left}.{sub_loc}"]]
                    _sub_loc_aggr_right += [inputs[f"{_prefix_right}.{sub_loc}"]]
                if len(rep.unit.split(".")) == 1:
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
            for _, rep in self.representations.items():
                _curr_source_ind_inc = _curr_source_ind + 1
                _prefix = (
                    f"source_{_curr_source_ind}->base"
                    if _curr_source_ind + 1 == len(self.representations)
                    else f"source_{_curr_source_ind}->source{_curr_source_ind_inc}"
                )
                _prefix_left = f"{_prefix}.0"
                _prefix_right = f"{_prefix}.1"
                _sub_loc_aggr_left = []  # 3d
                _sub_loc_aggr_right = []  # 3d
                for sub_loc in rep.unit.split("."):
                    _sub_loc_aggr_left += [inputs[f"{_prefix_left}.{sub_loc}"]]
                    _sub_loc_aggr_right += [inputs[f"{_prefix_right}.{sub_loc}"]]
                if len(rep.unit.split(".")) == 1:
                    _sub_loc_aggr_left = _sub_loc_aggr_left[0]
                    _sub_loc_aggr_right = _sub_loc_aggr_right[0]
                _curr_source_ind += 1
                batched_location_dict[_prefix] = (
                    [_sub_loc_aggr_left],  # 3D or 4D
                    [_sub_loc_aggr_right],  # 3D or 4D
                )

        return batched_location_dict
    
    def train(self, mode=True):
        self.model.train(mode=mode)
    
    def eval(self):
        self.model.eval()

    def train_alignment(
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

    def eval_alignment(
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


# Backwards-compatible alias: the dedicated nnsight/ndif backend has been merged
# into IntervenableModel, which now runs every model through nnsight.
IntervenableNdifModel = IntervenableModel


def build_intervenable_model(config, model, **kwargs):
    """
    Factory for intervenable models. All models — plain ``torch.nn.Module``s and
    pre-wrapped nnsight models alike — are driven by the single nnsight-backed
    :class:`IntervenableModel`.
    """
    return IntervenableModel(config, model, **kwargs)

