from typing import List, Tuple
from interface.detectors import Detector, Detection, Box2d
from interface.datasets import Sample, Size
from interface.datasets.detection import *
from interface.datasets.Batch import Batch
from interface.ITI import ITI
import torch
import cv2
import pycocotools.coco
import os
from interface.transforms import RandomCropAspectTransform, RandomRotateTransform, rotate, AutoContrast
from interface.transforms import ScaleTransform
import interface.transforms
dataDir = 'interface/datasets/coco'
dataType = 'val2014'
annFile = '%s/annotations/instances_%s.json' % (dataDir, dataType)

coco = pycocotools.coco.COCO(annFile)
coco.download("interface/datasets/coco/imgs", coco.getImgIds(catIds=[3]))


def show(t: torch.Tensor, wait: bool = False):
    if len(t.shape) == 3:
        t = t.unsqueeze(0)
    t = torch.nn.functional.interpolate(t, scale_factor=(1.0, 1.0))
    if len(t.shape) == 4:
        t = t[0]
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
        return cv2.waitKey(1)


device="cuda:0"
for itiName in ["VCAE6","DenseFuse"]:
    iti:ITI = ITI.named(itiName)().to(device)
    #iti:ITI = ITI.named("Identity")().to(device)
    itiNeedTraining=True
    if os.path.exists("iti_"+itiName+".pth"):
        try:
            iti.load_state_dict(torch.load("iti_"+itiName+".pth", map_location=device), strict=False)
            itiNeedTraining=False
        except:
            pass
    loss_fn = torch.nn.HuberLoss().to(device)
    preScale = ScaleTransform(640, 640)
    randomCrop = RandomCropAspectTransform(100,100,0.2,False)
    transform2 = ScaleTransform(480, 352)
    rotation = RandomRotateTransform([0,1,2,3,4,5,6,7,8,9,10,90,180,270,359,358,357,356,355,354,353,352,351,350])
    autoContrast = AutoContrast()
    transforms = [autoContrast,device,preScale,rotation,randomCrop,preScale]
    transforms = [autoContrast,device,rotation,randomCrop,preScale]

    if itiNeedTraining:
        optimizer = torch.optim.Adamax(iti.parameters(), lr=2e-4)

        dataset = CocoDetection("interface/datasets/coco/imgs", annFile)
        from tqdm import tqdm
        batch = Batch.of(dataset, 1)

        iter = int(5000/len(dataset))
        if iter ==0:
            iter=1
        for b in tqdm(range(iter)):
            for sample in tqdm(batch, leave=False):
                sample = interface.transforms.apply(sample,transforms)
                optimizer.zero_grad()
                output :Sample = iti(sample)
                loss = sum([ iti.loss(a, b) for (a,b) in zip(sample,output)])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                show(output[0].getRGB())
        torch.save(iti.state_dict(), "iti_"+itiName+".pth")
    else:
        print("Already trained")
