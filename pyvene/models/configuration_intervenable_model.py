import json, warnings, torch
from collections import OrderedDict, namedtuple
from typing import Any, List, Mapping, Optional

from transformers import PreTrainedTokenizer, TensorType, is_torch_available
from transformers.configuration_utils import PretrainedConfig

from .interventions import VanillaIntervention


IntervenableRepresentationConfig = namedtuple(
    "IntervenableRepresentationConfig",
    "intervenable_layer intervenable_representation_type "
    "intervenable_unit max_number_of_units "
    "intervenable_low_rank_dimension "
    "subspace_partition group_key intervention_link_key intervenable_moe "
    "source_representation hidden_source_representation",
    defaults=(
        0, "block_output", "pos", 1, 
        None, None, None, None, None, None, None),
)


class IntervenableConfig(PretrainedConfig):
    def __init__(
        self,
        intervenable_representations=[IntervenableRepresentationConfig()],
        intervenable_interventions_type=VanillaIntervention,
        mode="parallel",
        intervenable_interventions=[None],
        sorted_keys=None,
        intervention_dimensions=None,
        intervenable_model_type=None,
        **kwargs,
    ):
        if isinstance(intervenable_representations, list):
            self.intervenable_representations = intervenable_representations
        else:
            self.intervenable_representations = [intervenable_representations]
        self.intervenable_interventions_type = intervenable_interventions_type
        self.mode = mode
        self.intervenable_interventions = intervenable_interventions
        self.sorted_keys = sorted_keys
        self.intervention_dimensions = intervention_dimensions
        self.intervenable_model_type = intervenable_model_type
        super().__init__(**kwargs)

    def __repr__(self):
        intervenable_representations = []
        for reprs in self.intervenable_representations:
            if isinstance(reprs, list):
                reprs = IntervenableRepresentationConfig(*reprs)
            new_d = {}
            for k, v in reprs._asdict().items():
                if type(v) not in {str, int, list, tuple, dict} and v is not None and v != [None]:
                    new_d[k] = "PLACEHOLDER"
                else:
                    new_d[k] = v
            intervenable_representations += [new_d]
        _repr = {
            "intervenable_model_type": str(self.intervenable_model_type),
            "intervenable_representations": tuple(intervenable_representations),
            "intervenable_interventions_type": str(
                self.intervenable_interventions_type
            ),
            "mode": self.mode,
            "intervenable_interventions": [
                str(intervenable_intervention)
                for intervenable_intervention in self.intervenable_interventions
            ],
            "sorted_keys": tuple(self.sorted_keys) if self.sorted_keys is not None else str(self.sorted_keys),
            "intervention_dimensions": str(self.intervention_dimensions),
        }
        _repr_string = json.dumps(_repr, indent=4)

        return f"IntervenableConfig\n{_repr_string}"

    def __str__(self):
        return self.__repr__()
