import json, warnings, torch
from collections import OrderedDict, namedtuple
from typing import Any, List, Mapping, Optional

from transformers import PreTrainedTokenizer, TensorType, is_torch_available
from transformers.configuration_utils import PretrainedConfig

from .interventions import VanillaIntervention


RepresentationConfig = namedtuple(
    "RepresentationConfig",
    "layer component unit "
    "max_number_of_units "
    "low_rank_dimension intervention_type "
    "subspace_partition group_key intervention_link_key moe_key "
    "source_representation hidden_source_representation",
    defaults=(
        0, "block_output", "pos", 1, None,
        None, None, None, None, None, None, None),
)


class IntervenableConfig(PretrainedConfig):
    def __init__(
        self,
        representations=[RepresentationConfig()],
        intervention_types=VanillaIntervention,
        mode="parallel",
        interventions=[None],
        sorted_keys=None,
        model_type=None, # deprecating
        # hidden fields for backlog
        intervention_dimensions=None,
        **kwargs,
    ):
        if not isinstance(representations, list):
            representations = [representations]

        casted_representations = []
        for reprs in representations:
            if isinstance(reprs, RepresentationConfig):
                casted_representations += [reprs]
            elif isinstance(reprs, list):
                casted_representations += [
                    RepresentationConfig(*reprs)]
            elif isinstance(reprs, dict):
                casted_representations += [
                    RepresentationConfig(**reprs)]
            else:
                raise ValueError(
                    f"{reprs} format in our representation list is not supported.")
        self.representations = casted_representations
        self.intervention_types = intervention_types
        # the type inside reprs can overwrite
        overwrite = False
        overwrite_intervention_types = []
        for reprs in self.representations:
            if overwrite:
                if reprs.intervention_type is None:
                    raise ValueError(
                        "intervention_type if used should be specified for all")
            if reprs.intervention_type is not None:
                overwrite = True
                overwrite_intervention_types += [reprs.intervention_type]
        if None in overwrite_intervention_types:
            raise ValueError(
                "intervention_type if used should be specified for all")
        if overwrite:
            self.intervention_types = overwrite_intervention_types
            
        self.mode = mode
        self.interventions = interventions
        self.sorted_keys = sorted_keys
        self.intervention_dimensions = intervention_dimensions
        self.model_type = model_type
        super().__init__(**kwargs)
    
    def add_intervention(self, representations):
        if not isinstance(representations, list):
            representations = [representations]

        for reprs in representations:
            if isinstance(reprs, RepresentationConfig):
                self.representations += [reprs]
            elif isinstance(reprs, list):
                self.representations += [
                    RepresentationConfig(*reprs)]
            elif isinstance(reprs, dict):
                self.representations += [
                    RepresentationConfig(**reprs)]
            else:
                raise ValueError(
                    f"{reprs} format in our representation list is not supported.")
            if self.representations[-1].intervention_type is None:
                raise ValueError(
                    "intervention_type should be provided.")
            self.intervention_types += [self.representations[-1].intervention_type]
            
    def __repr__(self):
        representations = []
        for reprs in self.representations:
            if isinstance(reprs, list):
                reprs = IntervenableRepresentationConfig(*reprs)
            new_d = {}
            for k, v in reprs._asdict().items():
                if type(v) not in {str, int, list, tuple, dict} and v is not None and v != [None]:
                    new_d[k] = "PLACEHOLDER"
                else:
                    new_d[k] = v
            representations += [new_d]
        _repr = {
            "model_type": str(self.model_type),
            "representations": tuple(representations),
            "intervention_types": str(
                self.intervention_types
            ),
            "mode": self.mode,
            "interventions": [
                str(intervention)
                for intervention in self.interventions
            ],
            "sorted_keys": tuple(self.sorted_keys) if self.sorted_keys is not None else str(self.sorted_keys),
            "intervention_dimensions": str(self.intervention_dimensions),
        }
        _repr_string = json.dumps(_repr, indent=4)

        return f"IntervenableConfig\n{_repr_string}"

    def __str__(self):
        return self.__repr__()
