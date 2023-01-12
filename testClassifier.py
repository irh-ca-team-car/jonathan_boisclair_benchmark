from interface.classifiers.Classifier import Classifier, Classification
from interface.datasets.detection.A2 import A2Detection
from interface.datasets.Batch import Batch
import torch
import cv2

from interface.datasets.Sample import Sample
from interface.datasets.classification import ClassificationDataset

device ="cuda:0" 
for (name,clz) in Classifier.getAllRegisteredDetectors().items():
#for iti in [ITI.named("DenseFuse")]:
    model :Classifier = clz().to(device)
    dataset = ClassificationDataset.named("CIFAR-10")
    optimizer = torch.optim.Adamax(model.parameters(), lr=2e-4)
    loss_fn = torch.nn.HuberLoss().to(device)

    for e in range(100):
        for sample in Batch(dataset,4):
            print(sample)
            sample = [s.to(device) for s in sample]
            output = model.forward(sample)

            print([o.getCategoryName() for o in output])

            #print(output.getCategoryName())

