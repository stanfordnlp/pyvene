from abc import ABC, abstractmethod
import torch


class AlignableBase(torch.nn.Module, ABC):

    @abstractmethod
    def get_rotation_parameters(self):
        pass

    @abstractmethod
    def get_boundary_parameters(self):
        pass

    @abstractmethod
    def get_temperature(self):
        pass

    @abstractmethod
    def set_temperature(self, temp: torch.Tensor):
        pass
