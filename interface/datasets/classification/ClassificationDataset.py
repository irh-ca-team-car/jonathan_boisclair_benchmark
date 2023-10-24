

from typing import Dict, List, Tuple, Union
from ..Sample import Classification, Sample
import torch
class ClassificationDataset:
    def __init__(self) -> None:
        self.clz = []
        pass
    def classesList(self) -> List[str]:
        return list(self.clz)
    def getId(self,str: str):
        return 0
    def getName(self,id=None):
        return str(id)
    def isBanned(nameOrId):
        return False
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index: int) -> Sample:
        return Sample()

    datasets: Dict[str,"ClassificationDataset"] = dict()
    def register(name:Union["ClassificationDataset",str],dataset:Union[None,"ClassificationDataset"]=None):
        if isinstance(name, ClassificationDataset):
            name_ = name.getName()
            dataset = name
            name = name_
        ClassificationDataset.datasets[name]= dataset
    def named(name:str, name_=None) -> "ClassificationDataset":
        if isinstance(name,ClassificationDataset):
            return ClassificationDataset.named(name_)
        return ClassificationDataset.datasets[name]
    def registered() -> List[Tuple[str,"ClassificationDataset"]]:
        return list(ClassificationDataset.datasets.items())
