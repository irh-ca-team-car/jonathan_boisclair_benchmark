from typing import List, Tuple
from interface.datasets.detection.A2 import A2Detection
from interface.detectors import Detector
from interface.datasets import DetectionDataset,Detection, Sample, Size
from interface.datasets.Batch import Batch
from interface.ITI import ITI
from interface.metrics.Metrics import AveragePrecision, MultiImageAveragePrecision
import torch
from tqdm import tqdm
configs = [
    ("VCAE6","yolov5n"),
    ("Identity","yolov5n"),
    ("DenseFuse","yolov5n"),
    #("Identity","ssd"),
    #("Identity","fasterrcnn_resnet50_fpn"),
]

datasets : List[Tuple[str,DetectionDataset]] = [
    ("A2",A2Detection("/home/boiscljo/git/pytorch_ros/src/distributed/data/fusiondata/all.csv")),
    ("FLIR_CONVERTED",A2Detection("data/FLIR_CONVERTED/all.csv"))]
datasets_train : List[Tuple[str,DetectionDataset]] = [
    ("A2",A2Detection("/home/boiscljo/git/pytorch_ros/src/distributed/data/fusiondata/all.csv")),
    ("FLIR_CONVERTED",A2Detection("data/FLIR_CONVERTED/all.csv"))]
datasets_eval : List[Tuple[str,DetectionDataset]] = [
    ("A2",A2Detection("/home/boiscljo/git/pytorch_ros/src/distributed/data/fusiondata/all.csv")),
    ("FLIR_CONVERTED",A2Detection("data/FLIR_CONVERTED/all.csv"))]
def addCSV(dataset, iti, detector, mAP):
    line = f"{dataset},{iti},{detector},{float(mAP)}"
    file1 = open("SOTA_A2E.csv", "a") # append mode
    file1.write(f"{line}\n")
    file1.close()
    print(dataset, iti, detector, mAP)
import os
try:
    os.mkdir("a2e")
except:
    pass
device = "cuda:0"
for (name,dataset),(_,dataset_train),(_,dataset_eval) in zip(datasets,datasets_train,datasets_eval):
    for iti,detector in configs:
        print(name,iti,detector)
        detections = []
        ground_truths=[]

        iti_impl = ITI.named(iti)().to(device)
        if "CAE" in iti:
            #VCAE6_A2_retinanet_resnet50_fpn_v2.pth
            try:
                iti_impl.load_state_dict(torch.load("VCAE6_A2_retinanet_resnet50_fpn_v2.pth", map_location=device), strict=False)
            except:
                pass
        model = Detector.named(detector).to(device)
        #if "yolo" not in detector:
        model = model.adaptTo(dataset)

        if os.path.exists("a2e/"+detector+".pth"):
            try:
                model.load_state_dict(torch.load("a2e/"+detector+".pth", map_location=device), strict=False)
            except:
                pass
        else:
            model.train()
            optimizer = torch.optim.Adamax(model.parameters())
            epochs = tqdm(range(20), leave=False)
            for b in epochs:
                for cocoSamp in tqdm(Batch.of(datasets[1][1],16), leave=False):
                    cocoSamp =[samp.scale(Size(640,640)).to(device) for samp in cocoSamp]
                    with torch.no_grad():
                        values=iti_impl.forward(cocoSamp)
                    losses: torch.Tensor = (model.calculateLoss(values))
                    #loss_iti = sum([ iti_impl.loss(a, b) for (a,b) in zip(cocoSamp,values)])
                    #losses += loss_iti 
                    optimizer.zero_grad()
                    if not torch.isnan(losses):
                        losses.backward()
                        optimizer.step()
                    optimizer.zero_grad()
                    epochs.desc = str(losses.item())
                    losses = 0
            torch.save(model.state_dict(),"a2e/"+detector+".pth")

        #Do Pretraining
        #images
        if len(dataset.images) > 1000:
            dataset_train.images = dataset.images[0:300]
            dataset_eval.images = dataset.images[300:1000]
        else:
            dataset_train.images = dataset.images[0:int(len(dataset.images)/2)]
            dataset_eval.images = dataset.images[int(len(dataset.images)/2):]

        if True:
            model.train()
            optimizer = torch.optim.Adamax(model.parameters())
            epochs = tqdm(range(100), leave=False)
            for b in epochs:
                for cocoSamp in tqdm(Batch.of(dataset_train,1), leave=False):
                    cocoSamp[0]=cocoSamp[0].scale(Size(352,352)).to(device)
                    with torch.no_grad():
                        values=iti_impl.forward(cocoSamp)
                    losses: torch.Tensor = (model.calculateLoss(values))
                    #loss_iti = sum([ iti_impl.loss(a, b) for (a,b) in zip(cocoSamp,values)])
                    #losses += loss_iti 
                    optimizer.zero_grad()
                    if not torch.isnan(losses):
                        losses.backward()
                        optimizer.step()
                    optimizer.zero_grad()
                    epochs.desc = str(losses.item())
                    losses = 0

            model.eval()


        with torch.no_grad():
            for sample in tqdm(dataset_eval, leave=False, desc="Evaluating mAP"):
                sample = sample.scale(Size(640,640)).to(device)
                ground_truths.append(sample.detection)
                detected = model(iti_impl(sample))
                detections.append(detected)
        def filter_(classIdx):
            return classIdx <=5
        mAP = MultiImageAveragePrecision(ground_truths, detections)


        precisions = [AveragePrecision(x,y).precision(0.5) for (x,y) in zip(ground_truths,detections)]

        precision = sum(precisions) / len(precisions)
        #mAP.filter = filter_
        #mAP=mAP.mAP(0.01)

        addCSV(name,iti,detector,precision)


            

