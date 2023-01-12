

from typing import Dict, List, Tuple
from ..Sample import Classification
import torch
class ClassificationDataset:
    def __init__(self) -> None:
        pass
    def classesList() -> List[str]:
        return []
    def getId(str: str):
        return 0
    def getName(id=None):
        return str(id)
    def isBanned(nameOrId):
        return False
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index: int) -> Classification:
        return Classification(torch.Tensor())

    datasets: Dict[str,"ClassificationDataset"] = dict()
    def register(name:str,dataset:"ClassificationDataset", v=None):
        if isinstance(name,ClassificationDataset):
            return ClassificationDataset.register(dataset,v)
        ClassificationDataset.datasets[name]= dataset
    def named(name:str, name_=None) -> "ClassificationDataset":
        if isinstance(name,ClassificationDataset):
            return ClassificationDataset.named(name_)
        return ClassificationDataset.datasets[name]
    def registered() -> List[Tuple[str,"ClassificationDataset"]]:
        return list(ClassificationDataset.datasets.items())
