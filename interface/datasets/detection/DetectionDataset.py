from typing import List, Tuple, Union, Dict
from ..Sample import Sample

class DetectionDataset:
    def __init__(self) -> None:
        pass
    def classesList(self) -> List[str]:
        return []
    def getId(self,str: str):
        return 0
    def getName(self,id=None):
        return str(id)
    def isBanned(self,nameOrId):
        return False
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index: int) -> Sample:
        return Sample()

    def withMax(self,maxValue) -> "DetectionDataset":
        return self
    def withSkip(self,maxValue) -> "DetectionDataset":
        return self
    def shuffled(self,maxValue) -> "DetectionDataset":
        return self

    datasets: Dict[str,"DetectionDataset"] = dict()
    def register(name:Union["DetectionDataset",str],dataset:Union[None,"DetectionDataset"]=None):
        if isinstance(name, DetectionDataset):
            name_ = name.getName()
            dataset = name
            name = name_
        DetectionDataset.datasets[name]= dataset
    def named(name:str, name_=None) -> "DetectionDataset":
        if isinstance(name,DetectionDataset):
            return DetectionDataset.named(name_)
        return DetectionDataset.datasets[name]
    def registered() -> List[Tuple[str,"DetectionDataset"]]:
        return list(DetectionDataset.datasets.items())