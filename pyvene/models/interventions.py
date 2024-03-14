import torch
import numpy as np
from abc import ABC, abstractmethod

from .layers import RotateLayer, LowRankRotateLayer, SubspaceLowRankRotateLayer
from .basic_utils import sigmoid_boundary
from .intervention_utils import _can_use_fast, _do_intervention_by_swap


class Intervention(torch.nn.Module):

    """Intervention the original representations."""

    def __init__(self, **kwargs):
        super().__init__()
        self.trainable = False
        self.is_source_constant = False

        self.keep_last_dim = kwargs["keep_last_dim"] if "keep_last_dim" in kwargs else False
        self.use_fast = kwargs["use_fast"] if "use_fast" in kwargs else False
        self.subspace_partition = (
            kwargs["subspace_partition"] if "subspace_partition" in kwargs else None
        )
        # we turn the partition into list indices
        if self.subspace_partition is not None:
            expanded_subspace_partition = []
            for subspace in self.subspace_partition:
                if len(subspace) == 2 and isinstance(subspace[0], int):
                    expanded_subspace_partition.append([i for i in range(subspace[0],subspace[1])])
                else:
                    # it could be discrete indices.
                    expanded_subspace_partition.append(subspace)
            self.subspace_partition = expanded_subspace_partition
            
        if "embed_dim" in kwargs and kwargs["embed_dim"] is not None:
            self.register_buffer('embed_dim', torch.tensor(kwargs["embed_dim"]))
            self.register_buffer('interchange_dim', torch.tensor(kwargs["embed_dim"]))
        else:
            self.embed_dim = None
            self.interchange_dim = None
            
        if "source_representation" in kwargs and kwargs["source_representation"] is not None:
            self.is_source_constant = True
            self.register_buffer('source_representation', kwargs["source_representation"])
        else:
            if "hidden_source_representation" in kwargs and \
                kwargs["hidden_source_representation"] is not None:
                self.is_source_constant = True
            else:
                self.source_representation = None
                
    def set_source_representation(self, source_representation):
        self.is_source_constant = True
        self.register_buffer('source_representation', source_representation)
                
    def set_interchange_dim(self, interchange_dim):
        if isinstance(interchange_dim, int):
            self.interchange_dim = torch.tensor(interchange_dim)
        else:
            self.interchange_dim = interchange_dim
            
    @abstractmethod
    def forward(self, base, source, subspaces=None):
        pass


class LocalistRepresentationIntervention(torch.nn.Module):

    """Localist representation."""

    def __init__(self, **kwargs):
        super().__init__()
        self.is_repr_distributed = False
        
        
class DistributedRepresentationIntervention(torch.nn.Module):

    """Distributed representation."""

    def __init__(self, **kwargs):
        super().__init__()
        self.is_repr_distributed = True
        
        
class TrainableIntervention(Intervention):

    """Intervention the original representations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainable = True
        self.is_source_constant = False
        
    def tie_weight(self, linked_intervention):
        pass


class ConstantSourceIntervention(Intervention):

    """Constant source."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_source_constant = True


class SourcelessIntervention(Intervention):

    """No source."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_source_constant = True
    
    
class BasisAgnosticIntervention(Intervention):

    """Intervention that will modify its basis in a uncontrolled manner."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.basis_agnostic = True


class SharedWeightsTrainableIntervention(TrainableIntervention):

    """Intervention the original representations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shared_weights = True
        
        
class ZeroIntervention(ConstantSourceIntervention, LocalistRepresentationIntervention):

    """Zero-out activations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, base, source=None, subspaces=None):
        return _do_intervention_by_swap(
            base,
            torch.zeros_like(base),
            "interchange",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )

    def __str__(self):
        return f"ZeroIntervention()"
        
        
class CollectIntervention(ConstantSourceIntervention):

    """Collect activations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, base, source=None, subspaces=None):
        return _do_intervention_by_swap(
            base,
            source,
            "collect",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )

    def __str__(self):
        return f"CollectIntervention()"
        
        
class SkipIntervention(BasisAgnosticIntervention, LocalistRepresentationIntervention):

    """Skip the current intervening layer's computation in the hook function."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, base, source, subspaces=None):
        # source here is the base example input to the hook
        return _do_intervention_by_swap(
            base,
            source,
            "interchange",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )

    def __str__(self):
        return f"SkipIntervention()"


class VanillaIntervention(Intervention, LocalistRepresentationIntervention):

    """Intervention the original representations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, base, source, subspaces=None): 
        return _do_intervention_by_swap(
            base,
            source if self.source_representation is None else self.source_representation,
            "interchange",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )

    def __str__(self):
        return f"VanillaIntervention()"


class AdditionIntervention(BasisAgnosticIntervention, LocalistRepresentationIntervention):

    """Intervention the original representations with activation addition."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, base, source, subspaces=None):
        return _do_intervention_by_swap(
            base,
            source if self.source_representation is None else self.source_representation,
            "add",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )

    def __str__(self):
        return f"AdditionIntervention()"


class SubtractionIntervention(BasisAgnosticIntervention, LocalistRepresentationIntervention):

    """Intervention the original representations with activation subtraction."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, base, source, subspaces=None):
        
        return _do_intervention_by_swap(
            base,
            source if self.source_representation is None else self.source_representation,
            "subtract",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )

    def __str__(self):
        return f"SubtractionIntervention()"


class RotatedSpaceIntervention(TrainableIntervention, DistributedRepresentationIntervention):

    """Intervention in the rotated space."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        rotate_layer = RotateLayer(self.embed_dim)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)

    def forward(self, base, source, subspaces=None):
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)
        # interchange
        rotated_base = _do_intervention_by_swap(
            rotated_base,
            rotated_source,
            "interchange",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )
        # inverse base
        output = torch.matmul(rotated_base, self.rotate_layer.weight.T)
        return output.to(base.dtype)

    def __str__(self):
        return f"RotatedSpaceIntervention()"


class BoundlessRotatedSpaceIntervention(TrainableIntervention, DistributedRepresentationIntervention):

    """Intervention in the rotated space with boundary mask."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        rotate_layer = RotateLayer(self.embed_dim)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.intervention_boundaries = torch.nn.Parameter(
            torch.tensor([0.5]), requires_grad=True
        )
        self.temperature = torch.nn.Parameter(torch.tensor(50.0))
        self.intervention_population = torch.nn.Parameter(
            torch.arange(0, self.embed_dim), requires_grad=False
        )

    def get_boundary_parameters(self):
        return self.intervention_boundaries

    def get_temperature(self):
        return self.temperature

    def set_temperature(self, temp: torch.Tensor):
        self.temperature.data = temp

    def set_intervention_boundaries(self, intervention_boundaries):
        self.intervention_boundaries = torch.nn.Parameter(
            torch.tensor([intervention_boundaries]), requires_grad=True
        )
        
    def forward(self, base, source, subspaces=None):
        batch_size = base.shape[0]
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)
        # get boundary
        intervention_boundaries = torch.clamp(self.intervention_boundaries, 1e-3, 1)
        boundary_mask = sigmoid_boundary(
            self.intervention_population.repeat(batch_size, 1),
            0.0,
            intervention_boundaries[0] * int(self.embed_dim),
            self.temperature,
        )
        boundary_mask = (
            torch.ones(batch_size, device=base.device).unsqueeze(dim=-1) * boundary_mask
        )
        boundary_mask = boundary_mask.to(rotated_base.dtype)
        # interchange
        rotated_output = (
            1.0 - boundary_mask
        ) * rotated_base + boundary_mask * rotated_source
        # inverse output
        output = torch.matmul(rotated_output, self.rotate_layer.weight.T)
        return output.to(base.dtype)

    def __str__(self):
        return f"BoundlessRotatedSpaceIntervention()"


class SigmoidMaskRotatedSpaceIntervention(TrainableIntervention, DistributedRepresentationIntervention):

    """Intervention in the rotated space with boundary mask."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        rotate_layer = RotateLayer(self.embed_dim)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        # boundary masks are initialized to close to 1
        self.masks = torch.nn.Parameter(
            torch.tensor([100.0] * self.embed_dim), requires_grad=True
        )
        self.temperature = torch.nn.Parameter(torch.tensor(50.0))

    def get_boundary_parameters(self):
        return self.intervention_boundaries

    def get_temperature(self):
        return self.temperature

    def set_temperature(self, temp: torch.Tensor):
        self.temperature.data = temp

    def forward(self, base, source, subspaces=None):
        batch_size = base.shape[0]
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)
        # get boundary mask between 0 and 1 from sigmoid
        boundary_mask = torch.sigmoid(self.masks / self.temperature)

        boundary_mask = (
            torch.ones(batch_size, device=base.device).unsqueeze(dim=-1) * boundary_mask
        )
        boundary_mask = boundary_mask.to(rotated_base.dtype)
        # interchange
        rotated_output = (
            1.0 - boundary_mask
        ) * rotated_base + boundary_mask * rotated_source
        # inverse output
        output = torch.matmul(rotated_output, self.rotate_layer.weight.T)
        return output.to(base.dtype)

    def __str__(self):
        return f"SigmoidMaskRotatedSpaceIntervention()"


class SigmoidMaskIntervention(TrainableIntervention, LocalistRepresentationIntervention):

    """Intervention in the original basis with binary mask."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask = torch.nn.Parameter(
            torch.zeros(self.embed_dim), requires_grad=True)
        
        self.temperature = torch.nn.Parameter(torch.tensor(0.01))

    def get_temperature(self):
        return self.temperature

    def set_temperature(self, temp: torch.Tensor):
        self.temperature.data = temp

    def forward(self, base, source, subspaces=None):
        batch_size = base.shape[0]
        # get boundary mask between 0 and 1 from sigmoid
        mask_sigmoid = torch.sigmoid(self.mask / torch.tensor(self.temperature)) 
        
        # interchange
        intervened_output = (
            1.0 - mask_sigmoid
        ) * base + mask_sigmoid * source

        return intervened_output

    def __str__(self):
        return f"SigmoidMaskIntervention()"
    

class LowRankRotatedSpaceIntervention(TrainableIntervention, DistributedRepresentationIntervention):

    """Intervention in the rotated space."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)

    def forward(self, base, source, subspaces=None):
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)
        if subspaces is not None:
            if self.use_fast or _can_use_fast(subspaces):
                if self.subspace_partition is None:
                    sel_subspace_indices = subspaces[0]
                else:
                    sel_subspace_indices = []
                    for subspace in subspaces[0]:
                        sel_subspace_indices.extend(
                            self.subspace_partition[subspace]
                        )
                diff = rotated_source - rotated_base
                assert rotated_base.shape[0] == len(subspaces)
                batched_subspace = diff[..., sel_subspace_indices].unsqueeze(dim=1)
                batched_weights = self.rotate_layer.weight[..., sel_subspace_indices].T
                output = base + torch.matmul(batched_subspace, batched_weights).squeeze(
                    dim=1
                )
            else:
                assert self.subspace_partition is not None
                output = []
                diff = rotated_source - rotated_base
                assert rotated_base.shape[0] == len(subspaces)
                batched_subspace = []
                batched_weights = []
                for example_i in range(len(subspaces)):
                    # render subspace as column indices
                    sel_subspace_indices = []
                    for subspace in subspaces[example_i]:
                        sel_subspace_indices.extend(
                            self.subspace_partition[subspace]
                        )

                    LHS = diff[example_i, sel_subspace_indices].unsqueeze(dim=0)
                    RHS = self.rotate_layer.weight[..., sel_subspace_indices].T
                    batched_subspace += [LHS]
                    batched_weights += [RHS]
                batched_subspace = torch.stack(batched_subspace, dim=0)
                batched_weights = torch.stack(batched_weights, dim=0)
                output = base + torch.matmul(batched_subspace, batched_weights).squeeze(
                    dim=1
                )
        else:
            output = base + torch.matmul(
                (rotated_source - rotated_base), self.rotate_layer.weight.T
            )
        return output.to(base.dtype)

    def __str__(self):
        return f"LowRankRotatedSpaceIntervention()"


class PCARotatedSpaceIntervention(BasisAgnosticIntervention, DistributedRepresentationIntervention):
    """Intervention in the pca space."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pca = kwargs["pca"]
        pca_mean = kwargs["pca_mean"]
        pca_std = kwargs["pca_std"]
        self.pca_components = torch.nn.Parameter(
            torch.tensor(pca.components_, dtype=torch.float32), requires_grad=False
        )
        self.pca_mean = torch.nn.Parameter(
            torch.tensor(pca_mean, dtype=torch.float32), requires_grad=False
        )
        self.pca_std = torch.nn.Parameter(
            torch.tensor(pca_std, dtype=torch.float32), requires_grad=False
        )
        self.trainable = False

    def forward(self, base, source, subspaces=None):
        base_norm = (base - self.pca_mean) / self.pca_std
        source_norm = (source - self.pca_mean) / self.pca_std

        rotated_base = torch.matmul(base_norm, self.pca_components.T)  # B * D_R
        rotated_source = torch.matmul(source_norm, self.pca_components.T)
        # interchange
        rotated_base = _do_intervention_by_swap(
            rotated_base,
            rotated_source,
            "interchange",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
        )
        # inverse base
        output = torch.matmul(rotated_base, self.pca_components)  # B * D
        output = (output * self.pca_std) + self.pca_mean
        return output

    def __str__(self):
        return f"PCARotatedSpaceIntervention()"

class NoiseIntervention(ConstantSourceIntervention, LocalistRepresentationIntervention):
    """Noise intervention"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        rs = np.random.RandomState(1)
        prng = lambda *shape: rs.randn(*shape)
        noise_level = kwargs["noise_leve"] \
            if "noise_leve" in kwargs else 0.13462981581687927 
        self.register_buffer('noise', torch.from_numpy(
            prng(1, 4, self.embed_dim)))
        self.register_buffer('noise_level', torch.tensor(noise_level))
        
    def forward(self, base, source=None, subspaces=None):
        base[..., : self.interchange_dim] += self.noise * self.noise_level
        return base

    def __str__(self):
        return f"NoiseIntervention()"
    