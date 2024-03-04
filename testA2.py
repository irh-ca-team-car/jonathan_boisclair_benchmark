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
    #cv2.imshow("Image", np_)
    # for i in range(30):

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
from interface.transforms import FLIR_FIX
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

br = AutoContrast()
for cocoSamp in tqdm(dataset):
    cocoSamp:Sample = cocoSamp
    print("A")
    scaled = FLIR_FIX(fnThermal(br(cocoSamp)))

    tmp = (scaled.clone().getThermal()*255).byte()
    imgt = scaled.detection.onImage(tmp, colors=[(255,0,0)])

    tmp = (scaled.clone().getRGB()*255).byte()
    imgv = scaled.detection.onImage(tmp, colors=[(255,0,0)])

    a = ((imgt.int()+imgv.int()) /2).byte()
    img = torch.cat([imgt,imgv,a],2)
    print("A")

    mean = img[0].float().mean()
    if show(img):
        torchvision.io.write_jpeg(img,"image.jpg")
        torchvision.io.write_jpeg((scaled.clone().getRGB()*255).byte(),"image_no_box.jpg")
        
        img = scaled.detection.onImage((scaled.clone().getRGB()*255).byte(), colors=[(128,128,255)])
        torchvision.io.write_jpeg(img,"image_all_box.jpg")

        scaled.detection.boxes2d=random.choices(scaled.detection.boxes2d, k=int(len(scaled.detection.boxes2d)*0.6))
        img = scaled.detection.onImage((scaled.clone().getRGB()*255).byte(), colors=[(255,0,0)])
        torchvision.io.write_jpeg(img,"image_less_box.jpg")

       
        break
    pass
    print("-")


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
