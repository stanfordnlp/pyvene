from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

class TaskBase(ABC):
    @abstractmethod
    def prepare_dataloader(self, tokenizer, **kwargs) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        pass

