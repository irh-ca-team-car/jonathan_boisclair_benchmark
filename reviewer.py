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
datasetValid=dataset.withSkip(500).withMax(500)
dataset=dataset.withSkip(7).withMax(500)
br = AutoContrast()
for cocoSamp in tqdm(dataset):
    break
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


#dataset = CitiscapesDetection(suffix="8bit.png")
#dataset = A2Detection("/home/boiscljo/git/pytorch_ros/src/distributed/data/fusiondata/all.csv")
#models = [(name,det()) for (name,det) in Detector.getAllRegisteredDetectors().items()]
#print([name for (name,det) in models])
def append(text):
    with open('review2.txt', 'a') as file:
        file.write(str(text)+"\r\n")

models = [
     Detector.named("retinanet_resnet50_fpn_v2").adaptTo(dataset) for x in range(len(fixs))
]
t=tqdm(range(1000))
for epoch in t:
    for fix_id,fix in enumerate(fixs):
        model = models[fix_id].to("cuda:0")
        model.train()
        optimizer = torch.optim.Adamax(model.parameters())
    
        batch=Batch.of(dataset,1)
        for cocoSamp in tqdm(batch, leave=False):
            scaled = fix(fnThermal(br(cocoSamp[0])))
            tmp = (scaled.clone().getThermal()*255).byte()
            tmp = tmp.cpu().permute(1, 2, 0).numpy()
            tmp= cv2.applyColorMap(tmp, cv2.COLORMAP_HOT)
            tmp=cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
            tmp = torch.from_numpy(tmp).permute(2, 0, 1)
            imgt =tmp # scaled.detection.onImage(tmp, colors=[(0,255,0)])

            tmp = (scaled.clone().getRGB()*255).byte()
            imgv = tmp # scaled.detection.onImage(tmp, colors=[(0,255,0)])

            a = ((imgt.int()*7+imgv.int()*3) /10).byte()

            s=Sample()
            s.setImage(a)
            s.detection = scaled.detection

            losses =(model.calculateLoss(cocoSamp))
            t.desc= (str(losses))
            losses.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses=0

        from interface.metrics.Metrics import DatasetAveragePrecision
        model.eval()
        map = DatasetAveragePrecision(model,datasetValid, verbose = True)
      
        append("Calibration "+str(fix_id)+" Epoch "+str(epoch)+" "+str(float(map.mAP(0.000000001))))
        torch.save(model.state_dict(),"review_"+str(fix_id)+".pt")
        model = model.to("cpu")

