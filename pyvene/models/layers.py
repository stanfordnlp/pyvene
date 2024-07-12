from abc import ABCMeta, abstractmethod

import torch


class InverseRotateLayer(torch.nn.Module):
    """The inverse of a given `LinearLayer` module."""

    def __init__(self, lin_layer):
        super().__init__()
        self.lin_layer = lin_layer

    def forward(self, x):
        output = torch.matmul(x, self.lin_layer.weight.T)
        return output


class RotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, init_orth=True):
        super().__init__()
        weight = torch.empty(n, n)
        # we don't need init if the saved checkpoint has a nice
        # starting point already.
        # you can also study this if you want, but it is our focus.
        if init_orth:
            torch.nn.init.orthogonal_(weight)
        self.weight = torch.nn.Parameter(weight, requires_grad=True)

    def forward(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)


class LowRankRotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, m, init_orth=True):
        super().__init__()
        # n > m
        self.weight = torch.nn.Parameter(torch.empty(n, m), requires_grad=True)
        if init_orth:
            torch.nn.init.orthogonal_(self.weight)

    def forward(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)


class SubspaceLowRankRotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization with subspace."""

    def __init__(self, n, m, init_orth=True):
        super().__init__()
        # n > m
        self.weight = torch.nn.Parameter(torch.empty(n, m), requires_grad=True)
        if init_orth:
            torch.nn.init.orthogonal_(self.weight)

    def forward(self, x, l, r):
        return torch.matmul(x.to(self.weight.dtype), self.weight[:, l:r])


class AutoencoderLayerBase(torch.nn.Module, metaclass=ABCMeta):
    """An abstract base class that defines an interface of an autoencoder."""

    @abstractmethod
    def encode(self, x):
        ...

    @abstractmethod
    def decode(self, latent):
        ...


class AutoencoderLayer(AutoencoderLayerBase):
    """An autoencoder with a single-layer encoder and single-layer decoder."""
    def __init__(self, input_dim, latent_dim, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = torch.nn.Sequential(
              torch.nn.Linear(input_dim, latent_dim, bias=True),
              torch.nn.ReLU())
        self.decoder = torch.nn.Sequential(
              torch.nn.Linear(latent_dim, input_dim, bias=True))

    def encode(self, x):
        x = x.to(self.encoder[0].weight.dtype)
        x = x - self.decoder[0].bias
        latent = self.encoder(x)
        return latent

    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, base, return_latent=False):
        base_type = base.dtype
        base = base.to(self.encoder[0].weight.dtype)
        latent = self.encode(base)
        base_reconstruct = self.decode(latent)
        if not return_latent:
            return base_reconstruct.to(base_type)
        return {'latent': latent, 'output': base_reconstruct}
