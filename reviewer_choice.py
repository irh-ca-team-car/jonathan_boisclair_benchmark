import cv2
cv2.namedWindow("Image", cv2.WINDOW_KEEPRATIO)

from interface.detectors import Detector, Detection, Box2d
from interface.datasets import Sample, Size
from interface.datasets.detection.A2 import A2Detection
from interface.datasets.Batch import Batch
import interface
import torch
import time
import torchvision
print(__file__)

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
    k = cv2.waitKey(1)
    # for i in range(30):
    #k = cv2.waitKey(1)

    while wait:
        cv2.imshow("Image", np_)
        k = cv2.waitKey(1)
        if k == 27:  # Esc key to stop
            return False
        if k == 115:
            return True
    return False
print(__file__)
dataset = A2Detection("data/FLIR_CONVERTED/all.csv")
print(__file__)
from tqdm import tqdm
import random
from interface.transforms.Scale import scale
from interface.transforms.TorchVisionFunctions import AdjustBrightness, AutoContrast
br = AdjustBrightness(0.6)

def fnThermal(sample: Sample):
    thermal = sample.getThermal()
    min = thermal.min()
    max = thermal.max()
    range = max-min
    thermal = (thermal-min)/range
    sample.setThermal(thermal)
    return sample

print(__file__)
def FLIR_FIX(sample:Sample):
        x,y,w,h=27, 74, 649, 690

        if isinstance(sample,list):
            return [FLIR_FIX(x) for x in sample]
        sample = scale(sample,400,400)
        rgb = sample.getRGB()
        tmp = torch.nn.functional.interpolate(rgb.unsqueeze(0), size=(h,w)).squeeze(0)[:,(y):(y+400),(x):(x+400)]
        sample.setImage(tmp)
        return sample
def FLIR_FIX2(sample:Sample):
        x,y,w,h=100, 54, 599, 490

        if isinstance(sample,list):
            return [FLIR_FIX2(x) for x in sample]
        sample = scale(sample,400,400)
        rgb = sample.getRGB()
        tmp = torch.nn.functional.interpolate(rgb.unsqueeze(0), size=(h,w)).squeeze(0)[:,(y):(y+400),(x):(x+400)]
        sample.setImage(tmp)
        return sample
def FLIR_FIX3(sample:Sample):
        x,y,w,h=52, 49, 469, 490

        if isinstance(sample,list):
            return [FLIR_FIX3(x) for x in sample]
        sample = scale(sample,400,400)
        rgb = sample.getRGB()
        tmp = torch.nn.functional.interpolate(rgb.unsqueeze(0), size=(h,w)).squeeze(0)[:,(y):(y+400),(x):(x+400)]
        sample.setImage(tmp)
        return sample
def FLIR_FIX4(sample:Sample):
        x,y,w,h=32, 49, 449, 500

        if isinstance(sample,list):
            return [FLIR_FIX4(x) for x in sample]
        sample = scale(sample,400,400)
        rgb = sample.getRGB()
        tmp = torch.nn.functional.interpolate(rgb.unsqueeze(0), size=(h,w)).squeeze(0)[:,(y):(y+400),(x):(x+400)]
        sample.setImage(tmp)
        return sample
def FLIR_FIX5(sample:Sample):
        x,y,w,h=37, 44, 449, 490

        if isinstance(sample,list):
            return [FLIR_FIX5(x) for x in sample]
        sample = scale(sample,400,400)
        rgb = sample.getRGB()
        tmp = torch.nn.functional.interpolate(rgb.unsqueeze(0), size=(h,w)).squeeze(0)[:,(y):(y+400),(x):(x+400)]
        sample.setImage(tmp)
        return sample
fixs=[
   FLIR_FIX,
   FLIR_FIX2,
   FLIR_FIX3,
   FLIR_FIX4,
   FLIR_FIX5
]
dataset=dataset.withSkip(7)
br = AutoContrast()
for i,cocoSamp in enumerate(tqdm(dataset)):
    l = []
    for fix in fixs:
        cocoSamp:Sample = cocoSamp
        scaled = fix(fnThermal(br(cocoSamp)))

        tmp = (scaled.clone().getThermal()*255).byte()
        tmp = tmp.cpu().permute(1, 2, 0).numpy()
        tmp= cv2.applyColorMap(tmp, cv2.COLORMAP_HOT)
        tmp=cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        tmp = torch.from_numpy(tmp).permute(2, 0, 1)
        imgt =tmp # scaled.detection.onImage(tmp, colors=[(0,255,0)])

        tmp = (scaled.clone().getRGB()*255).byte()
        imgv = tmp # scaled.detection.onImage(tmp, colors=[(0,255,0)])

        a = ((imgt.int()*7+imgv.int()*3) /10).byte()
        img = torch.cat([imgt,imgv,a],2)
        l.append(a)
        
    show(torch.cat(l,2), False)
    torchvision.io.write_jpeg(torch.cat(l,2),"imgs/"+str(i))

