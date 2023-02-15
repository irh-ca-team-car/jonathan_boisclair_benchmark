from typing import Dict, List, Tuple
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
from interface.metrics.Metrics import AveragePrecision
dataDir = 'interface/datasets/coco'
dataType = 'val2014'
annFile = '%s/annotations/instances_%s.json' % (dataDir, dataType)

coco = pycocotools.coco.COCO(annFile)
coco.download("interface/datasets/coco/imgs", coco.getImgIds(catIds=[3]))

datasets : List[Tuple[str,DetectionDataset]] = [
    ("A1_UQTR_REGULAR",A1Detection("data/attention-data/UQTRR/full.txt")),
    ("A2",A2Detection("/home/boiscljo/git/pytorch_ros/src/distributed/data/fusiondata/all.csv")),
    ("FLIR_CONVERTED",A2Detection("data/FLIR_CONVERTED/all.csv")),
    ("CitiscapesDetection_8bit",CitiscapesDetection(mode="train", suffix="8bit.png")),
    ("CitiscapesDetection_0.005",CitiscapesDetection(mode="train", suffix="0.005.png")),
    ("CitiscapesDetection_0.01",CitiscapesDetection(mode="train", suffix="0.01.png")),
    ("CitiscapesDetection_0.02",CitiscapesDetection(mode="train", suffix="0.02.png")),
    ("CitiscapesDetection_8bitval",CitiscapesDetection(mode="val", suffix="8bit.png")),
    ("CitiscapesDetection_0.005val",CitiscapesDetection(mode="val", suffix="0.005.png")),
    ("CitiscapesDetection_0.01val",CitiscapesDetection(mode="val", suffix="0.01.png")),
    ("CitiscapesDetection_0.02val",CitiscapesDetection(mode="val", suffix="0.02.png")),
    ("CocoDetection",CocoDetection("interface/datasets/coco/imgs", annFile))
]

device="cuda:0"

def removeCSV(dataset):
    file = f"mAP_{dataset}.csv"
    if os.path.exists(file):
        os.unlink(file)
def addCSV(dataset, image_id, iti, performances ):
    line = f"{image_id},{iti},{','.join([str(x) for x in performances])}"
    file1 = open(f"mAP_{dataset}.csv", "a") # append mode
    file1.write(f"{line}\n")
    file1.close()
preScale = ScaleTransform(640, 640)
with torch.no_grad():
    for dname,dataset in datasets:
        from tqdm import tqdm
        dataset = dataset.withMax(1000)
        models : List[Tuple[str,Detector]] = [
            ("yolov5n",Detector.named("yolov5n")),
            ("yolov5s",Detector.named("yolov5s")),
            ("yolov5m",Detector.named("yolov5m")),
            ("retinanet_resnet50_fpn_v2",Detector.named("retinanet_resnet50_fpn_v2"))
        ]
        
        nets = []
        for (mname, det) in models:
            for itiName in ["Identity","VCAE6"]:
                iti = ITI.named(itiName)().to(device)
                iti.name = itiName

                model: Detector = det.adaptTo(dataset).to(device)
                save_name = itiName+"_"+dname+"_"+mname+".pth"
                tmpModule = torch.nn.ModuleList([model,iti])
                if os.path.exists(save_name):
                    try:
                        tmpModule.load_state_dict(torch.load(save_name, map_location=device), strict=False)
                    except:
                        raise Exception(save_name+" does not load")
                else:
                    raise Exception(save_name+" does not load")
               
                nets.append((mname,iti,model))
        removeCSV(dname)
        names=[]
        for name,iti,det in nets:
            if not name in names:
                names.append(name)
        addCSV(dname,"image_id","iti", names)

        t = tqdm(dataset, leave=True, desc = dname)
        for b, samp in enumerate(t):
            performances:Dict[str,List]={}
            for i, (mname,iti, det) in enumerate(nets):
                if not iti.name in performances:
                    performances[iti.name]=[]
                detection = det(iti(preScale(samp)))
                performances[iti.name].append(AveragePrecision(samp.detection, detection).pascal())
                
            for key,value in performances.items():
                addCSV(dname,str(b),key, value)
                
            


    