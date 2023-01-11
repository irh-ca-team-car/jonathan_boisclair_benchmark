from interface.detectors import Detector, Detection, Box2d
from interface.datasets import Sample, Size
from interface.datasets.Citiscapes import CitiscapesDetection
from interface.datasets.A1 import A1Detection
from interface.datasets.A2 import A2Detection
from interface.datasets.Coco import CocoDetection
from interface.datasets.Batch import Batch
from interface.datasets.OpenImages import OpenImagesDetection
from interface.impl import EfficientDetector
from interface.ITI import ITI
import interface
import torch
import time
import cv2
import torchvision
import torchvision.transforms as transforms
import pycocotools.coco
import fiftyone.zoo as foz
import os

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
        cv2.waitKey(1)


datasets = [
    #("A1_UQTR_REGULAR",A1Detection("data/attention-data/UQTRR/full.txt")),
    #("A2",A2Detection("/home/boiscljo/git/pytorch_ros/src/distributed/data/fusiondata/all.csv")),
    ("FLIR_CONVERTED",A2Detection("data/FLIR_CONVERTED/all.csv")),
    # ("OpenImages",OpenImagesDetection(dataset=foz.load_zoo_dataset("open-images-v6",
    #                                                                      split="validation",
    #                                                                      max_samples=1000,
    #                                                                      seed=51,
    #                                                                      shuffle=False,
    #                                                                      label_type="detection",
    #                                                                      classes=[
    #                                                                          "Car"],
    #                                                                      dataset_name="openimagescar"
    #                                                                      ))),
    # CitiscapesDetection(mode="train", suffix="8bit.png"),
    #CitiscapesDetection(mode="train", suffix="0.005.png"),
    #CitiscapesDetection(mode="train", suffix="0.01.png"),
    #CitiscapesDetection(mode="train", suffix="0.02.png"),
    # CitiscapesDetection(mode="val", suffix="8bit.png"),
    #CitiscapesDetection(mode="val", suffix="0.005.png"),
    #CitiscapesDetection(mode="val", suffix="0.01.png"),
    #CitiscapesDetection(mode="val", suffix="0.02.png"),
    #CocoDetection("interface/datasets/coco/imgs", annFile)
]
#dataset = CitiscapesDetection(suffix="8bit.png")
#dataset = CitiscapesDetection(suffix="0.02.png")
device="cuda:0"
itiName = "VCAE6"
itiName = "DenseFuse"
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
itiNeedTraining=False
if itiNeedTraining:
    optimizer = torch.optim.Adamax(iti.parameters(), lr=2e-4)

    for dname,dataset in datasets:
        from tqdm import tqdm
        batch = Batch.of(dataset, 4)

        iter = int(500/len(dataset))
        if iter ==0:
            iter=1
        for b in range(iter):
            for sample in tqdm(batch):
                sample = [c.scale(Size(480, 352)).to(device) for c in sample]
                optimizer.zero_grad()
                output :Sample = iti(sample)

                loss = sum([ iti.loss(a, b) for (a,b) in zip(sample,output)])
                #loss = loss_fn(sample.getRGB(), output.getRGB())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                show(output[0].getRGB())
    torch.save(iti.state_dict(), "iti_"+itiName+".pth")

for dname,dataset in datasets:
    from tqdm import tqdm
    # models = [(name, det())
    #      for (name, det) in Detector.getAllRegisteredDetectors().items()]
    #models = [models[-1]]
    #models = [("EfficientDetector_d0", Detector.named("EfficientDetector_d0"))]
    models = [("retinanet_resnet50_fpn_v2",Detector.named("retinanet_resnet50_fpn_v2"))]
    print([name for (name, det) in models])

    for i, (name, det) in enumerate(models):
        model: Detector = det.adaptTo(dataset.__class__).to(device)
        
        
        model.train()
        tmpModule = torch.nn.ModuleList([model,iti])
        optimizer = torch.optim.Adamax(tmpModule.parameters(), lr=6e-4)
        losses = 0
        batch = Batch.of(dataset, 1)

        save_name = itiName+"_"+dname+"_"+name+".pth"
        if os.path.exists(save_name):
            try:
                tmpModule.load_state_dict(torch.load(save_name, map_location=device), strict=False)
            except:
                pass

        for b, cocoSamp in enumerate(tqdm(batch)):
            #cocoSamp = [c.scale(Size(512, 416)).to(device) for c in cocoSamp]
            #cocoSamp = [c.scale(Size(752, 480)).to(device) for c in cocoSamp]
            cocoSamp = [c.scale(Size(480, 352)).to(device) for c in cocoSamp]
            
            if True:# dataset.__class__.getName() != "MS-COCO":
                values=iti.forward(cocoSamp)
                losses: torch.Tensor = (model.calculateLoss(values))
                loss_iti = sum([ iti.loss(a, b) for (a,b) in zip(cocoSamp,values)])
                losses += loss_iti 
                optimizer.zero_grad()
                if not torch.isnan(losses):
                    losses.backward()
                    tqdm.write(str(losses.item()))
                    optimizer.step()
                optimizer.zero_grad()
                losses = 0
            

            model.eval()

            cocoSamp_ = cocoSamp[-1]
            values = values[-1]
            del cocoSamp
            cocoSamp :Sample = cocoSamp_
            detections = model.forward(cocoSamp, dataset=dataset.__class__)
            workImage = values.clone()
            workImage = cocoSamp.detection.onImage(
                workImage, colors=[(255, 0, 0)])
            #workImage = detections.filter(0.1).onImage(workImage)
            workImage = detections.filter(0.3).NMS_Pytorch().onImage(workImage)
            #workImage = detections.filter(0.90).onImage(workImage)
            # for b in detections.boxes2d:
            #    print(b)
            show(workImage, False)
            #show(cocoSamp.detection.onImage(cocoSamp), False)
            model.train()
            cv2.waitKey(100)
            #del model

        torch.save(tmpModule.state_dict(),save_name)
        pass
