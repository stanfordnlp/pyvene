import torch
from abc import ABC, abstractmethod

from models.layers import RotateLayer, LowRankRotateLayer
from models.utils import sigmoid_boundary
        

class Intervention(torch.nn.Module, ABC):

    """Intervention the original representations."""
    def __init__(self):
        super().__init__()
        self.trainble = False
        
    @abstractmethod
    def set_interchange_dim(self, interchange_dim):
        pass

    @abstractmethod
    def forward(self, base, source):
        pass
    
    
class TrainbleIntervention(Intervention):

    """Intervention the original representations."""
    def __init__(self):
        super().__init__()
        self.trainble = True


class BasisAgnosticIntervention(Intervention):

    """Intervention that will modify its basis in a uncontrolled manner."""
    def __init__(self):
        super().__init__()
        self.basis_agnostic = True

        
class VanillaIntervention(Intervention):
    
    """Intervention the original representations."""
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        self.interchange_dim = None
        self.embed_dim = embed_dim

    def set_interchange_dim(self, interchange_dim):
        self.interchange_dim = interchange_dim

    def forward(self, base, source):
        # interchange
        base[:self.interchange_dim] = source[:self.interchange_dim]

        return base

    def __str__(self):
        return f"VanillaIntervention(embed_dim={self.embed_dim})"

    
class AdditionIntervention(BasisAgnosticIntervention):
    
    """Intervention the original representations with activation addition."""
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        self.interchange_dim = None
        self.embed_dim = embed_dim

    def set_interchange_dim(self, interchange_dim):
        self.interchange_dim = interchange_dim

    def forward(self, base, source):
        # interchange
        base[:self.interchange_dim] += source[:self.interchange_dim]

        return base

    def __str__(self):
        return f"AdditionIntervention(embed_dim={self.embed_dim})"
    

class SubstractionIntervention(BasisAgnosticIntervention):
    
    """Intervention the original representations with activation substraction."""
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        self.interchange_dim = None
        self.embed_dim = embed_dim

    def set_interchange_dim(self, interchange_dim):
        self.interchange_dim = interchange_dim

    def forward(self, base, source):
        # interchange
        base[:self.interchange_dim] -= source[:self.interchange_dim]

        return base

    def __str__(self):
        return f"SubstractionIntervention(embed_dim={self.embed_dim})"
    
    
class RotatedSpaceIntervention(TrainbleIntervention):
    
    """Intervention in the rotated space."""
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        rotate_layer = RotateLayer(embed_dim)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.interchange_dim = None
        self.embed_dim = embed_dim
        
    def set_interchange_dim(self, interchange_dim):
        self.interchange_dim = interchange_dim

    def forward(self, base, source):
        assert self.interchange_dim is not None
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)
        # interchange
        rotated_base[..., :self.interchange_dim] = rotated_source[..., :self.interchange_dim]
        # inverse base
        output = torch.matmul(rotated_base, self.rotate_layer.weight.T)
        return output.to(base.dtype)
    
    def __str__(self):
        return f"RotatedSpaceIntervention(embed_dim={self.embed_dim})"
    
    
class BoundlessRotatedSpaceIntervention(TrainbleIntervention):
    
    """Intervention in the rotated space with boundary mask."""
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        rotate_layer = RotateLayer(embed_dim)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(
            rotate_layer)

        self.intervention_boundaries = torch.nn.Parameter(
            torch.tensor([0.5]), requires_grad=True)
        self.temperature = torch.nn.Parameter(torch.tensor(50.0)) 
        self.embed_dim = embed_dim
        self.intervention_population = torch.nn.Parameter(
            torch.arange(0, self.embed_dim), requires_grad=False)
        
    def get_boundary_parameters(self):
        return self.intervention_boundaries

    def get_temperature(self):
        return self.temperature

    def set_temperature(self, temp: torch.Tensor):
        self.temperature.data = temp
        
    def set_interchange_dim(self, interchange_dim):
        """interchange dim is learned and can not be set"""
        assert False

    def forward(self, base, source):
        batch_size = base.shape[0]
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)
        # get boundary
        intervention_boundaries = torch.clamp(
            self.intervention_boundaries, 1e-3, 1)
        boundary_mask = sigmoid_boundary(
            self.intervention_population.repeat(batch_size, 1), 
            0.,
            intervention_boundaries[0] * int(self.embed_dim),
            self.temperature
        )
        boundary_mask = torch.ones(
            batch_size, device=base.device).unsqueeze(dim=-1)*boundary_mask
        boundary_mask = boundary_mask.to(rotated_base.dtype)
        # interchange
        rotated_output = (1. - boundary_mask)*rotated_base + boundary_mask*rotated_source
        # inverse output
        output = torch.matmul(rotated_output, self.rotate_layer.weight.T)
        return output.to(base.dtype)
    
    def __str__(self):
        return f"BoundlessRotatedSpaceIntervention(embed_dim={self.embed_dim})"
    
    
class LowRankRotatedSpaceIntervention(TrainbleIntervention):
    
    """Intervention in the rotated space."""
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        rotate_layer = LowRankRotateLayer(embed_dim, kwargs["proj_dim"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.interchange_dim = None
        self.embed_dim = embed_dim
        
    def set_interchange_dim(self, interchange_dim):
        self.interchange_dim = interchange_dim

    def forward(self, base, source):
        rotated_base = self.rotate_layer(base)
        low_rank_approx_base = base - torch.matmul(rotated_base, self.rotate_layer.weight.T)
        rotated_source = self.rotate_layer(source)
        # interchange
        inv_value = rotated_source
        # inverse base
        output = torch.matmul(inv_value, self.rotate_layer.weight.T) + low_rank_approx_base
        return output.to(base.dtype)
    
    def __str__(self):
        return f"LowRankRotatedSpaceIntervention(embed_dim={self.embed_dim})"
    
    
class PCARotatedSpaceIntervention(RotatedSpaceIntervention):
    
    """Intervention in the rotated space."""
    def __init__(self, embed_dim, **kwargs):
        super().__init__(embed_dim)
        self.pca_components = torch.nn.Parameter(torch.tensor(
            pca.components_, dtype=torch.float32), requires_grad=False)
        self.pca_mean = torch.nn.Parameter(
            torch.tensor(pca_mean, dtype=torch.float32), requires_grad=False)
        self.pca_std = torch.nn.Parameter(
            torch.tensor(pca_std, dtype=torch.float32), requires_grad=False)
        self.interchange_dim = 10 # default to be 10.
        self.embed_dim = embed_dim
        self.trainble = False
        
    def set_interchange_dim(self, interchange_dim):
        self.interchange_dim = interchange_dim
        
    def forward(self, base, source):
        base_norm = (base - self.pca_mean) / self.pca_std
        source_norm = (source - self.pca_mean) / self.pca_std

        rotated_base = torch.matmul(base_norm, self.pca_components.T)  # B * D_R
        rotated_source = torch.matmul(source_norm, self.pca_components.T)
        dims = list(range(self.interchange_dim))
        rotated_base[:, dims] = rotated_source[:, dims]
        # inverse base
        output = torch.matmul(rotated_base, self.pca_components)  # B * D
        output = (output * self.pca_std) + self.pca_mean
        return output
    
    def __str__(self):
        return f"PCARotatedSpaceIntervention(embed_dim={self.embed_dim})"
    
