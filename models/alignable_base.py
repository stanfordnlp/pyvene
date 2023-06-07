from abc import ABC, abstractmethod
from collections.abc import Sequence
import torch


class AlignableBase(torch.nn.Module, ABC):

    @abstractmethod
    def get_rotation_parameters(self) -> Sequence[torch.Tensor]:
        pass

    @abstractmethod
    def get_boundary_parameters(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_temperature(self) -> torch.Tensor:
        pass

    @abstractmethod
    def set_temperature(self, temp: torch.Tensor):
        pass
