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
    

    def classesList(self) -> List[str]:
        
        return list(self.lazy_load().classes)
    def getId(self,str: str):
        return self.lazy_load().classes.index(str)
    def getName(self,id):
        return self.lazy_load().classes[id]
    def lazy_load(self):
        if self.fodataset is None:
            self.fodataset = fiftyone.zoo.load_zoo_dataset(self.name,**self.lazy)
            self.data = list(self.fodataset)
            self.classes = self.fodataset.get_classes("classification")
        return self
    def __init__(self, name:str, **kwargs) -> None:
        super().__init__()
        self.name = name
        self.lazy = kwargs
        self.fodataset = None
        self.data = None
        self.classes = None
        
    def __len__(self):
        return len(self.lazy_load().fodataset)

    def convert(self,data:fiftyone.Sample)-> Sample:
        s = Sample()
        s.setImage(torchvision.io.read_image(data.to_dict()["filepath"]))
        s.classification = Classification(self.getId(data.to_dict()["ground_truth"]["label"]),self)
        return s
    def __getitem__(self, index: int) -> Classification:
        foSamples = self.lazy_load().data[index]
        print(foSamples)
        print(self.__len__)

        if isinstance(foSamples,list):
            return [self.convert(x) for x in foSamples]
        return self.convert(foSamples)

ClassificationDataset.register("CIFAR-10[train]", FiftyOneDataset("cifar10", split="train"))
ClassificationDataset.register("CIFAR-10[test]", FiftyOneDataset("cifar10", split="test"))
ClassificationDataset.register("CIFAR-10[validation]", FiftyOneDataset("cifar10", split="validation"))
ClassificationDataset.register("CIFAR-10", FiftyOneDataset("cifar10"))

ClassificationDataset.register("CIFAR-100[train]", FiftyOneDataset("cifar100", split="train"))
ClassificationDataset.register("CIFAR-100[test]", FiftyOneDataset("cifar100", split="test"))
ClassificationDataset.register("CIFAR-100", FiftyOneDataset("cifar100"))

ClassificationDataset.register("Caltech-256", FiftyOneDataset("caltech256"))