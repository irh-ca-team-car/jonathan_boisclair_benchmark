from interface.detectors import Detector, Detection, Box2d
from interface.datasets import Sample, Size
from interface.datasets.detection.A2 import A2Detection
from interface.datasets.detection.Coco import CocoDetection
from interface.datasets.Batch import Batch
import interface
import torch
import time
import cv2
import torchvision

x,y,w,h=37, 44, 449, 490

def show(t: torch.Tensor,wait: bool = False):
    global x,y,w,h
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

    while True:
            #right = 83
            #left = 81
            #up = 82
            #down = 84
            #a=97
            #s=115
            #d=100
            #w=119
            cv2.imshow("Image", np_)
            k = cv2.waitKey(1)
            if k == 83:
                x = x+1
            if k == 81:
                x = x-1
            if k == 82:
                y = y-1
            if k == 84:
                y = y+1
            if k == 97:
                w = w-1
            if k == 100:
                w = w+1
            if k == 119:
                h = h+1
            if k == 115:
                h = h+1
            if w<400:
                w=400
            if h<400:
                h=400

            if k == 27:
                return True
            return False
dataset = A2Detection("data/FLIR_CONVERTED/all.csv")
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

def FLIR_FIX(sample:Sample):
    global x,y,w,h
    if isinstance(sample,list):
        return [FLIR_FIX(x) for x in sample]
    sample = scale(sample,400,400)
    rgb = sample.getRGB()
    if x + 400 > w:
        x = w-400
    if y + 400 > h:
        y = h-400
    tmp = torch.nn.functional.interpolate(rgb.unsqueeze(0), size=(h,w)).squeeze(0)[:,(y):(y+400),(x):(x+400)]
    sample.setImage(tmp)

    print(x,y,w,h)
    return sample

br = AutoContrast()

sample = dataset[0]

while True:
    scaled = FLIR_FIX(fnThermal(br(sample)))

    tmp = (scaled.clone().getThermal()*255).byte()
    imgt = scaled.detection.onImage(tmp, colors=[(255,0,0)])
    tmp = (scaled.clone().getRGB()*255).byte()
    imgv = scaled.detection.onImage(tmp, colors=[(255,0,0)])
    a = ((imgt.int()+imgv.int()) /2).byte()
    img = torch.cat([imgt,imgv,a],2)
    if show(img):
        break

exit(0)

#dataset = CitiscapesDetection(suffix="8bit.png")
#dataset = A2Detection("/home/boiscljo/git/pytorch_ros/src/distributed/data/fusiondata/all.csv")
dataset = A2Detection("data/FLIR_CONVERTED/all.csv")
from tqdm import tqdm
#models = [(name,det()) for (name,det) in Detector.getAllRegisteredDetectors().items()]
models=[("model",list(Detector.getAllRegisteredDetectors().items())[0][1]())]
#print([name for (name,det) in models])
for i,(name,det) in enumerate(models):
    model :Detector = det.adaptTo(dataset).to("cuda:0")
    model.train()
    optimizer = torch.optim.Adamax(model.parameters())
    losses = 0
    batch=Batch.of(dataset,1)

    #dataset.images= dataset.images[int(2512*2):]
    #print(dataset.images[0])
    for cocoSamp in tqdm(batch):
        pass
        # cocoSamp = [c.scale(Size(512,400)) for c in cocoSamp]
        # losses +=(model.calculateLoss(cocoSamp))

        # optimizer.zero_grad()
        # losses.backward()
        # optimizer.step()
        # optimizer.zero_grad()
        # losses=0

        # model.eval()

        # cocoSamp_=cocoSamp[-1]
        # del cocoSamp
        # cocoSamp=cocoSamp_
        # detections = model.forward(cocoSamp,dataset= A2Detection)
        # workImage = cocoSamp.clone()
        # workImage = cocoSamp.detection.onImage(workImage, colors=[(255,0,0)])
        # workImage = detections.filter(0.5).onImage(workImage)
        # show(workImage, False)
        # #show(cocoSamp.detection.onImage(cocoSamp), False)
        # model.train()
        # cv2.waitKey(33)
        #del model
    pass
