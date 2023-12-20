import json, warnings
from collections import OrderedDict, namedtuple
from typing import Any, List, Mapping, Optional 

from transformers import PreTrainedTokenizer, TensorType, is_torch_available
from transformers.configuration_utils import PretrainedConfig

from models.interventions import VanillaIntervention


AlignableRepresentationConfig = namedtuple(
    "AlignableRepresentationConfig", 
    "alignable_layer alignable_representation_type "\
    "alignable_unit max_number_of_units "\
    "alignable_low_rank_dimension "\
    "subspace_partition group_key intervention_link_key",
    defaults=(0, "block_output", "pos", 1, None, None, None, None)
)


class AlignableConfig(PretrainedConfig):
    def __init__(
        self,
        alignable_model_type="gpt2",
        alignable_representations=[
            AlignableRepresentationConfig()
        ],
        alignable_interventions_type=VanillaIntervention,
        mode="parallel",
        alignable_interventions=[None],
        **kwargs
    ):
        self.alignable_model_type = alignable_model_type
        self.alignable_representations = alignable_representations
        self.alignable_interventions_type = alignable_interventions_type
        self.mode = mode
        self.alignable_interventions = alignable_interventions
        super().__init__(**kwargs)
    
    def __repr__(self):
        _repr = {
            "alignable_model_type": str(self.alignable_model_type),
            "alignable_representations": tuple(self.alignable_representations),
            "alignable_interventions_type": str(self.alignable_interventions_type),
            "mode": self.mode,
            "alignable_interventions": [
                str(alignable_intervention) \
                for alignable_intervention in alignable_interventions
            ]
        }
        _repr_string = json.dumps(_repr, indent=4)
        
        return f"AlignableConfig\n{_repr_string}"

