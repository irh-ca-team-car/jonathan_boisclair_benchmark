from typing import List
import torch
from interface.datasets import Sample, Size, Detection
from interface.datasets.Batch import Batch
from interface.adapters.OpenCV import CVAdapter
from torchvision.transforms import ToPILImage
from interface.datasets.detection.A2 import A2Detection
import cv2
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True, trust_repo=True, autoshape=True).cpu()

print(model.__class__)

from interface.detectors.YoloV5 import YoloV5Detector

model = YoloV5Detector().to("cpu").train()
dataset = A2Detection("data/FLIR_CONVERTED/all.csv")

optimizer = torch.optim.Adamax(model.parameters(), lr=2e-3)

for samp in Batch.of(dataset,4):
    optimizer.zero_grad()
    loss = (model.calculateLoss(samp))
    print(loss)
    if not torch.isnan(loss):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    with torch.no_grad():
        result:List[Detection] = model(samp)
        Sample.show(result[0].onImage(samp[0]),False)
        cv2.waitKey(1)



exit(0)
# Images
imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images ,'https://ultralytics.com/images/zidane.jpg'
imgs = [ToPILImage()(Sample.Example().scale(Size(640,640)).getRGB())]
#print(imgs.shape)
# Inference
results = model(imgs)

# Results
#results.print()
results.show() 
print(results.__class__.__name__)
#print(results[0].__class__.__name__)
exit(0)
results.xyxy[0]  # img1 predictions (tensor)
print(results.pandas().xyxy[0])  # img1 predictions (pandas)
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie