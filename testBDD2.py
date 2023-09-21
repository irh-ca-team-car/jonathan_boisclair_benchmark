import sklearn
import fiftyone
from interface.detectors import Detector, Detection, Box2d
from interface.datasets import Sample, Size,Segmentation
from interface.datasets.detection.BDD import BDDDetection
from interface.datasets.Batch import Batch
import interface
import torch
import time
import cv2
import torchvision
import random

dataset = BDDDetection(classes=3)
print(len(dataset))

s1 = dataset[15]
img = s1.segmentation.colored()
img = s1.segmentation.detection.onImage(img).int()
rgb = (s1.getRGB()*255.0).int()
img = ((img+rgb) /2).byte()
torchvision.io.write_png(img,"colored_s0.png")


from interface.segmentation.Segmenter import Segmenter
from interface.datasets.Sample import Sample
import torch
from interface.transforms.Scale import ScaleTransform
scale = ScaleTransform(384,256)

device = "cuda:0"
mdl_name = "unet+mit_b0"
model_1 = Segmenter.named(mdl_name).adaptTo(dataset).to(device)
def mdl():
    return model_1
name = mdl_name
model_ctr = mdl
model = model_1
optim: torch.optim.Optimizer = model.optimizer(model,2e-3,0) 
lambda_group1 = lambda epoch: (5+(random.random()*2-1)) *10** (-(6+float(epoch) / 1000 - float(epoch) // 1000))
lambda_group2 = lambda epoch: (5+(random.random()*2-1)) *10** (-(10+float(epoch) / 1000- float(epoch) // 1000))

try:
    model.load_state_dict(torch.load("car/lanes2.pth", map_location=device), strict=False)
except:
    pass
for i in range(100):
    if i>=5:
        model.unfreeze_backbone()
    else:
        model.freeze_backbone()
    optim.param_groups[0]["lr"] = lambda_group1(i)
    optim.param_groups[1]["lr"] = lambda_group2(i)
    for b in Batch.of(dataset, 16):
        optim.zero_grad()
        b = scale(b)
        loss = (model.calculateLoss(b))
        loss.backward()

        m: Segmentation = model.forward(b)[0]
        
        torchvision.io.write_png((m.onImage(b[0])*255.0).byte(),"progress2.png")

        optim.step()
        optim.zero_grad()
        torch.save(model.state_dict(), "car/lanes2.pth")
