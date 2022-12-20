from interface.detectors import Detector
from interface.detectors import Sample
import interface
import torch

for i,(name,det) in enumerate(Detector.getAllRegisteredDetectors().items()):
    print(name,det)
    sample  = Sample()
    model = det()
    detections = model.forward(sample)
    print(detections)
    detections = model.forward([sample,sample])
    print(detections)