from typing import List, Tuple
from interface.datasets.detection.A2 import A2Detection
from interface.detectors import Detector
from interface.datasets import DetectionDataset,Detection, Sample, Size
from interface.datasets.Batch import Batch
from interface.ITI import ITI
from interface.metrics.Metrics import AveragePrecision, MultiImageAveragePrecision
from interface.transforms import apply, FLIR_FIX
from interface.transforms.Scale import ScaleTransform
import torch
from tqdm import tqdm
configs = [
    #("VCAE6","yolov5n"),
    #("Identity","yolov5n"),
    #("DenseFuse","yolov5n"),
    ("VCAE6","yolov7"),
    ("Identity","yolov7"),
    ("DenseFuse","yolov7"),
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
        model = Detector.named(detector)
        #if "yolo" not in detector:
        model = model.adaptTo(dataset).to(device)

        if os.path.exists("a2e/"+detector+".pth"):
            try:
                model.load_state_dict(torch.load("a2e/"+detector+".pth", map_location=device), strict=False)
            except:
                pass
        if True:
            model.train()
            optimizer = torch.optim.Adamax(model.parameters())
            epochs = tqdm(range(100), leave=False)
            for b in epochs:
                inner = tqdm(Batch.of(datasets[1][1].withMax(180),2), leave=False)
                for cocoSamp in inner:
                    model.train()
                    cocoSamp=apply(cocoSamp,[FLIR_FIX,"cuda:0"])
                    values= cocoSamp
                    for r in range(3):
                        losses: torch.Tensor = (model.calculateLoss(values))
                        #loss_iti = sum([ iti_impl.loss(a, b) for (a,b) in zip(cocoSamp,values)])
                        #losses += loss_iti 
                        optimizer.zero_grad()
                        if not torch.isnan(losses):
                            losses.backward()
                            optimizer.step()
                        optimizer.zero_grad()
                    inner.desc = str(losses.item())
                    losses = 0
                    model.eval()

                    cocoSamp_ = cocoSamp[0]
                    del cocoSamp
                    cocoSamp :Sample = cocoSamp_
                    detections = model.forward(cocoSamp, dataset=dataset)
                    workImage = cocoSamp.clone()
                    workImage = cocoSamp.detection.onImage(
                        workImage, colors=[(255, 0, 0)])
                    detections=detections.filter(0.15)

                    workImage = detections.NMS_Pytorch().onImage(workImage, colors=[(128, 128, 255)])

                    Sample.show(workImage,False, "pretaining of "+detector)

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
                    model.train()
                    cocoSamp=apply(cocoSamp,[FLIR_FIX,"cuda:0",ScaleTransform(Size(352,352))])
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

            cocoSamp_ = cocoSamp[0]
            del cocoSamp
            cocoSamp :Sample = cocoSamp_
            detections = model.forward(cocoSamp, dataset=dataset)
            workImage = cocoSamp.clone()
            workImage = cocoSamp.detection.onImage(
                        workImage, colors=[(255, 0, 0)])
            detections=detections.filter(0.3)

            workImage = detections.NMS_Pytorch().onImage(workImage, colors=[(128, 128, 255)])

            Sample.show(workImage,False, "Training of "+detector)



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


            

