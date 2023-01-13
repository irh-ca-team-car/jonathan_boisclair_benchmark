from typing import List
from .ClassificationDataset import ClassificationDataset
from ..Sample import Sample, Classification
import torchvision
import fiftyone
import fiftyone.zoo 
class FiftyOneDataset(ClassificationDataset):
    fodataset: fiftyone.Dataset
    data : List[fiftyone.Sample]
    classes : List[str]
    name: str
    a3name: str

    def classesList(self) -> List[str]:
        
        return list(self.lazy_load().classes)
    def getId(self,str: str):
        return self.lazy_load().classes.index(str)
    def getName(self,id=None):
        if id is None:
            return self.a3name
        return self.lazy_load().classes[id]
    def lazy_load(self):
        if self.fodataset is None:
            self.fodataset = fiftyone.zoo.load_zoo_dataset(self.name,**self.lazy)
            self.data = list(self.fodataset)
            self.classes = self.fodataset.get_classes("classification")
        return self
    def __init__(self, a3name:str, name:str, **kwargs) -> None:
        super().__init__()
        self.a3name =a3name
        self.name = name
        self.lazy = kwargs
        self.fodataset = None
        self.data = None
        self.classes = None
        
    def __len__(self):
        return len(self.lazy_load().fodataset)

    def convert(self,data:fiftyone.Sample)-> Sample:
        s = Sample()
        s.setImage(torchvision.io.read_image(data.to_dict()["filepath"]).float()/255.0)
        s.classification = Classification(self.getId(data.to_dict()["ground_truth"]["label"]),self)
        return s
    def __getitem__(self, index: int) -> Classification:
        foSamples = self.lazy_load().data[index]
        if isinstance(foSamples,list):
            return [self.convert(x) for x in foSamples]
        return self.convert(foSamples)

FiftyOneDataset("CIFAR-1","cifar10", split="train", max_samples=1).register()

FiftyOneDataset("CIFAR-10[train]","cifar10", split="train").register()
FiftyOneDataset("CIFAR-10[test]","cifar10", split="test").register()
FiftyOneDataset("CIFAR-10[validation]","cifar10", split="validation").register()
FiftyOneDataset("CIFAR-10","cifar10").register()

FiftyOneDataset("CIFAR-100[train]","cifar100", split="train").register()
FiftyOneDataset("CIFAR-100[test]","cifar100", split="test").register()
FiftyOneDataset("CIFAR-100","cifar100").register()

FiftyOneDataset("Caltech-256","caltech256").register()