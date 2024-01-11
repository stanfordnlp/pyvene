import torch
from abc import ABC, abstractmethod

from pyvene.models.layers import RotateLayer, LowRankRotateLayer, SubspaceLowRankRotateLayer
from pyvene.models.basic_utils import sigmoid_boundary
from pyvene.models.intervention_utils import _do_intervention_by_swap


class Intervention(torch.nn.Module):

    """Intervention the original representations."""

    def __init__(self, **kwargs):
        super().__init__()
        self.trainble = False
        self.use_fast = kwargs["use_fast"] if "use_fast" in kwargs else False

    @abstractmethod
    def set_interchange_dim(self, interchange_dim):
        pass

    @abstractmethod
    def forward(self, base, source):
        pass


class TrainableIntervention(Intervention):

    """Intervention the original representations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainble = True

    def tie_weight(self, linked_intervention):
        pass


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


class SkipIntervention(BasisAgnosticIntervention):

    """Skip the current intervening layer's computation in the hook function."""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.interchange_dim = embed_dim  # assuming full subspace
        self.subspace_partition = (
            kwargs["subspace_partition"] if "subspace_partition" in kwargs else None
        )

    def set_interchange_dim(self, interchange_dim):
        self.interchange_dim = interchange_dim

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
        return f"SkipIntervention(embed_dim={self.embed_dim})"


class VanillaIntervention(Intervention):

    """Intervention the original representations."""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.interchange_dim = embed_dim  # assuming full subspace
        self.subspace_partition = (
            kwargs["subspace_partition"] if "subspace_partition" in kwargs else None
        )

    def set_interchange_dim(self, interchange_dim):
        self.interchange_dim = interchange_dim

    def forward(self, base, source, subspaces=None):
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
        return f"VanillaIntervention(embed_dim={self.embed_dim})"


class AdditionIntervention(BasisAgnosticIntervention):

    """Intervention the original representations with activation addition."""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.interchange_dim = embed_dim  # assuming full subspace
        self.subspace_partition = (
            kwargs["subspace_partition"] if "subspace_partition" in kwargs else None
        )

    def set_interchange_dim(self, interchange_dim):
        self.interchange_dim = interchange_dim

    def forward(self, base, source, subspaces=None):
        return _do_intervention_by_swap(
            base,
            source,
            "add",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )

    def __str__(self):
        return f"AdditionIntervention(embed_dim={self.embed_dim})"


class SubtractionIntervention(BasisAgnosticIntervention):

    """Intervention the original representations with activation subtraction."""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.interchange_dim = embed_dim  # assuming full subspace
        self.subspace_partition = (
            kwargs["subspace_partition"] if "subspace_partition" in kwargs else None
        )

    def set_interchange_dim(self, interchange_dim):
        self.interchange_dim = interchange_dim

    def forward(self, base, source, subspaces=None):
        return _do_intervention_by_swap(
            base,
            source,
            "subtract",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )

    def __str__(self):
        return f"SubtractionIntervention(embed_dim={self.embed_dim})"


class RotatedSpaceIntervention(TrainableIntervention):

    """Intervention in the rotated space."""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        rotate_layer = RotateLayer(embed_dim)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.embed_dim = embed_dim
        self.interchange_dim = embed_dim  # assuming full subspace
        self.subspace_partition = (
            kwargs["subspace_partition"] if "subspace_partition" in kwargs else None
        )

    def set_interchange_dim(self, interchange_dim):
        self.interchange_dim = interchange_dim

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
        return f"RotatedSpaceIntervention(embed_dim={self.embed_dim})"


class BoundlessRotatedSpaceIntervention(TrainableIntervention):

    """Intervention in the rotated space with boundary mask."""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        rotate_layer = RotateLayer(embed_dim)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.subspace_partition = (
            kwargs["subspace_partition"] if "subspace_partition" in kwargs else None
        )
        # TODO: in case there are subspace partitions, we
        #       need to initialize followings differently.
        self.intervention_boundaries = torch.nn.Parameter(
            torch.tensor([0.5]), requires_grad=True
        )
        self.temperature = torch.nn.Parameter(torch.tensor(50.0))
        self.embed_dim = None
        self.interchange_dim = embed_dim  # assuming full subspace
        self.intervention_population = torch.nn.Parameter(
            torch.arange(0, self.embed_dim), requires_grad=False
        )

    def get_boundary_parameters(self):
        return self.intervention_boundaries

    def get_temperature(self):
        return self.temperature

    def set_temperature(self, temp: torch.Tensor):
        self.temperature.data = temp

    def set_interchange_dim(self, interchange_dim):
        """interchange dim is learned and can not be set"""
        assert False

    def forward(self, base, source, subspace=None):
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
        return f"BoundlessRotatedSpaceIntervention(embed_dim={self.embed_dim})"


class SigmoidMaskRotatedSpaceIntervention(TrainableIntervention):

    """Intervention in the rotated space with boundary mask."""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        rotate_layer = RotateLayer(embed_dim)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.subspace_partition = (
            kwargs["subspace_partition"] if "subspace_partition" in kwargs else None
        )
        # TODO: in case there are subspace partitions, we
        #       need to initialize followings differently.

        # boundary masks are initialized to close to 1
        self.masks = torch.nn.Parameter(
            torch.tensor([100] * embed_dim), requires_grad=True
        )
        self.temperature = torch.nn.Parameter(torch.tensor(50.0))
        self.embed_dim = embed_dim

    def get_boundary_parameters(self):
        return self.intervention_boundaries

    def get_temperature(self):
        return self.temperature

    def set_temperature(self, temp: torch.Tensor):
        self.temperature.data = temp

    def set_interchange_dim(self, interchange_dim):
        """interchange dim is learned and can not be set"""
        assert False

    def forward(self, base, source, subspace=None):
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
        return f"SigmoidMaskRotatedSpaceIntervention(embed_dim={self.embed_dim})"


class LowRankRotatedSpaceIntervention(TrainableIntervention):

    """Intervention in the rotated space."""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        rotate_layer = LowRankRotateLayer(embed_dim, kwargs["proj_dim"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.embed_dim = embed_dim
        self.interchange_dim = None
        self.subspace_partition = (
            kwargs["subspace_partition"] if "subspace_partition" in kwargs else None
        )

    def set_interchange_dim(self, interchange_dim):
        self.interchange_dim = interchange_dim

    def forward(self, base, source, subspaces=None):
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)
        if subspaces is not None:
            if self.use_fast:
                if self.subspace_partition is None:
                    sel_subspace_indices = subspaces[0]
                else:
                    sel_subspace_indices = []
                    for subspace in subspaces[0]:
                        sel_subspace_indices.extend(
                            [
                                i
                                for i in range(
                                    self.subspace_partition[subspace][0],
                                    self.subspace_partition[subspace][1],
                                )
                            ]
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
                            [
                                i
                                for i in range(
                                    self.subspace_partition[subspace][0],
                                    self.subspace_partition[subspace][1],
                                )
                            ]
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
        return f"LowRankRotatedSpaceIntervention(embed_dim={self.embed_dim})"


class PCARotatedSpaceIntervention(BasisAgnosticIntervention):
    """Intervention in the pca space."""

    def __init__(self, embed_dim, **kwargs):
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
        self.interchange_dim = 10  # default to be 10.
        self.embed_dim = embed_dim
        self.trainble = False
        self.subspace_partition = (
            kwargs["subspace_partition"] if "subspace_partition" in kwargs else None
        )

    def set_interchange_dim(self, interchange_dim):
        self.interchange_dim = interchange_dim

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
        return f"PCARotatedSpaceIntervention(embed_dim={self.embed_dim})"
