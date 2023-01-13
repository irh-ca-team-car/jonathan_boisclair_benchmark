from typing import List
from interface.classifiers.Classifier import Classifier, Classification
from interface.datasets.detection.A2 import A2Detection
from interface.datasets.Batch import Batch
import torch
import cv2

from interface.datasets.Sample import Sample, Size
from interface.datasets.classification import ClassificationDataset
from interface import transforms
from interface.adapters.OpenCV import VideoCapture, CVAdapter
import tqdm
device ="cuda:0" 
class CustomDataset(ClassificationDataset):
        def __init__(self) -> None:
            super().__init__()
            self.vid = VideoCapture(0)
        def __len__(self):
            return 1
        def classesList(self) -> List[str]:
            return [str(i) for i in range(5)]
        def __getitem__(self, index: int) -> Sample:
            if isinstance(index,slice):
                return [self.__getitem__(x) for x in [i for i in range(1)][index]]
            if index > 1:
                raise StopIteration()
            s = self.vid.__next__()
            s.classification = Classification(0,self)
            return s
dataset = CustomDataset()

def show(t: torch.Tensor,wait: bool = False):
    if len(t.shape) ==3:
        t=t.unsqueeze(0)
    t = torch.nn.functional.interpolate(t, scale_factor=(1.0,1.0))
    if len(t.shape) ==4:
        t=t[0]
    t = t.cpu().permute(1, 2, 0)
    np_ = t.detach().numpy()
    np_ = cv2.cvtColor(np_, cv2.COLOR_BGR2RGB)
    cv2.imshow("Image", np_)
    # for i in range(30):

    if wait:
        while True:
            cv2.imshow("Image", np_)
            k = cv2.waitKey(1)
            if k == 27:  # Esc key to stop
                break
    else:
        cv2.waitKey(1)
for (name,clz) in Classifier.getAllRegisteredDetectors().items():
#for iti in [ITI.named("DenseFuse")]:
    model :Classifier = clz().to(device).adaptTo(dataset)
    optimizer = torch.optim.Adamax(model.parameters(), lr=2e-2)
    loss_fn = torch.nn.HuberLoss().to(device)
    t=tqdm.tqdm(range(100),desc = name)
    for e in t:
        for sample in Batch(dataset,4):
            show(sample[0].getRGB())
            sample = transforms.scale(sample, Size(224,224) )
            sample = [s.to(device) for s in sample]
            optimizer.zero_grad()
            output = model.forward(sample)
            loss = model.calculateLoss(sample)
            t.desc = name +" " +str(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            

            #print(output.getCategoryName())

