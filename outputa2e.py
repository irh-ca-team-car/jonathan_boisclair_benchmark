from typing import List, Tuple
import cv2
import fiftyone
from interface.datasets.detection.A2 import A2Detection
from interface.datasets.detection.A2W import A2W
from interface.datasets.detection.PST900 import PST900Detection
from interface.detectors import Detector
from interface.datasets import DetectionDataset,Detection, Sample, Size
from interface.datasets.Batch import Batch
from interface.ITI import ITI
from interface.metrics.Metrics import AveragePrecision, MultiImageAveragePrecision
from interface.transforms import apply, FLIR_FIX
from interface.transforms.RandomCut import RandomCropAspectTransform
from interface.transforms.RandomRotate import RandomRotateTransform
from interface.transforms.Scale import ScaleTransform
from interface.classifiers import Classifier
import torch
from tqdm import tqdm

from interface.transforms.TorchVisionFunctions import AutoContrast
import socket
hostname = socket.gethostname()
if hostname == "irh-xavier":
    configs = [
        ("Identity","RBGT_A2_DET_vgg_8"),
        ("Identity","RBGT_A2_DET_alexnet_8"),
        ("Identity","RBGT_A2_DET_cae_8"),
    ]
    b_size = 64
else:
    configs = [
        #("VCAE6","yolov8n"),
        ("Identity","yolov8n"),
        #("DenseFuse","yolov8n"),
        #("VCAE6","yolov5n"),
        ("Identity","yolov5n"),
        #("DenseFuse","yolov5n"), 
        #("VCAE6","fasterrcnn_resnet50_fpn"),
        #("Identity","fasterrcnn_resnet50_fpn"),
        #("DenseFuse","fasterrcnn_resnet50_fpn"),
        #("VCAE6","ssd_lite"),
        #("Identity","ssd_lite"),
        #("DenseFuse","ssd_lite"),
    ]
    b_size = 8

datasets : List[Tuple[str,DetectionDataset]] = [
    ("PST900", PST900Detection()),
    ("A2",(A2Detection("/home/boiscljo/git/pytorch_ros/src/distributed/data/fusiondata/all.csv"))),
    ("FLIR_CONVERTED",(A2Detection("data/FLIR_CONVERTED/all.csv").withMax(180))),
    ]
datasets_train : List[Tuple[str,DetectionDataset]] = [
    ("PST900", PST900Detection()),
    ("A2",(A2Detection("/home/boiscljo/git/pytorch_ros/src/distributed/data/fusiondata/all.csv"))),
    ("FLIR_CONVERTED",(A2Detection("data/FLIR_CONVERTED/all.csv").withMax(180))),
    ]
datasets_eval : List[Tuple[str,DetectionDataset]] = [
    ("PST900", PST900Detection()),
    ("A2",(A2Detection("/home/boiscljo/git/pytorch_ros/src/distributed/data/fusiondata/all.csv"))),
    ("FLIR_CONVERTED",(A2Detection("data/FLIR_CONVERTED/all.csv").withMax(180))),
    ]
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

preScale = ScaleTransform(640, 640)
randomCrop = RandomCropAspectTransform(600,600,0.2,True)
transform2 = ScaleTransform(480, 352)
rotation = RandomRotateTransform([*range(0,10),*range(350,360)])
autoContrast = AutoContrast()
transforms = [autoContrast,preScale,rotation,randomCrop,preScale]
for (name,dataset),(_,dataset_train),(_,dataset_eval) in zip(datasets,datasets_train,datasets_eval):
    for iti,detector in configs:
        print(name,iti,detector)
        detections_ = []
        ground_truths=[]
        iti_impl = ITI.named(iti)
        iti_impl = iti_impl()
        iti_impl=iti_impl.to(device, dtype=torch.float32)
        for p in iti_impl.parameters():
            p.requires_grad_(False)
        if "CAE" in iti:
            #VCAE6_A2_retinanet_resnet50_fpn_v2.pth
            try:
                iti_impl.load_state_dict(torch.load("VCAE6_A2_retinanet_resnet50_fpn_v2.pth", map_location=device), strict=False)
            except:
                pass
        model = Detector.named(detector)
        #if "yolo" not in detector:
        model = model.adaptTo(dataset).to(device)
        if "A2" in detector:
            if "vgg" in detector:
                model.best_fit(Classifier.named("vgg11"))
            if "alexnet" in "detector":
                model.best_fit(Classifier.named("alexnet"))
        need_pretrain=True
        if os.path.exists("a2e/"+detector+".pth"):
            try:
                model.load_state_dict(torch.load("a2e/"+detector+".pth", map_location=device), strict=False)
                need_pretrain=False
            except:
                pass
        # try:
        #     model.freeze_backbone()
        # except:
        #     tqdm.write("Could not freeze backbone, training whole model")
        if need_pretrain:
            model.train()
            optimizer = model.optimizer(model)
            epochs = tqdm(range(400), leave=False)
            for b in epochs:
                dts = datasets[2][1].withMax(180)
                bts = Batch.of(dts,b_size)
                model.dataset=dts
                inner = tqdm(bts, leave=False)
                k=0
                for cocoSamp in inner:
                    model.train()
                    maybeFlir = []
                    if "A2" in name:
                        maybeFlir.append(FLIR_FIX)
                    cocoSamp=apply(cocoSamp,[*maybeFlir,device,*transforms])
                    for r in tqdm(range(1),leave=False):
                        losses: torch.Tensor = (model.calculateLoss(cocoSamp))
                        optimizer.zero_grad()
                        if not torch.isnan(losses):
                            losses.backward()
                            optimizer.step()
                        optimizer.zero_grad()
                    
                    inner.desc = str(losses.item())
                    del losses
                    model.eval()

                    cocoSamp_ = cocoSamp[0]
                    del cocoSamp
                    cocoSamp :Sample = cocoSamp_
                    detections = model.forward(cocoSamp, dataset=dataset)
                    workImage = cocoSamp.clone()
                    workImage = cocoSamp.detection.onImage(
                        workImage, colors=[(255, 0, 0)])
                    del cocoSamp
                    detections=detections.filter(0.05)

                    workImage = detections.NMS_Pytorch().onImage(workImage, colors=[(128, 128, 255)])
                    del detections
                    k=Sample.show(workImage,False, "pretaining of "+detector)
                    if k == 27:
                        exit()
                    del workImage
                torch.save(model.state_dict(),"a2e/"+detector+".pth")
                if k == 115:
                    break

        #Do Pretraining
        #images
        if len(dataset.images) > 1000:
            dataset_train.images = dataset.images[0:300]
            #dataset_eval.images = dataset.images[0:300]
            dataset_eval.images = dataset.images[300:1000]
        else:
            dataset_train.images = dataset.images[0:int(len(dataset.images)/2)]
            #dataset_eval.images = dataset.images[0:int(len(dataset.images)/2)]
            dataset_eval.images = dataset.images[int(len(dataset.images)/2):]
        need_fine_tune=True
        all_weights = torch.nn.ModuleDict({"iti":iti_impl,"detector":model})
        if os.path.exists("a2e/"+detector+"_"+name+".fine.pth"):
            try:
                # torch.save(model.state_dict(),"a2e/"+detector+"_"+name+".fine.pth")
                #torch.save(iti_impl.state_dict(),"a2e/"+iti+"_"+name+".fine.pth")
                model.load_state_dict(torch.load("a2e/"+detector+"_"+name+".pth", map_location=device), strict=False)
                iti_impl.load_state_dict(torch.load("a2e/"+iti+"_"+name+".pth", map_location=device), strict=False)
                #need_fine_tune=False
                pass
            except:
                pass
        if need_fine_tune:
            tqdm.write("Fine tuning for 500 epochs")
            model.train()
            try:
                model.freeze_backbone()
            except:
                tqdm.write("Could not freeze backbone, training whole model")
            optimizer = model.optimizer(model)
            #optimizer_iti = model.optimizer(iti_impl)
            #epochs = tqdm(range(500), leave=False)
            epochs = tqdm(range(500), leave=False)
            for b in epochs:
                l =0 
                mb=tqdm(Batch.of(dataset_train,b_size/2), leave=False)
                for cocoSamp in mb:
                    model.train()
                    optimizer.zero_grad()
                    #optimizer_iti.zero_grad()
                    maybeFlir=[]
                    if "A2" in name:
                        maybeFlir.append(FLIR_FIX)
                    cocoSamp=apply(cocoSamp,[*maybeFlir,device,preScale])
                    values=iti_impl.forward(cocoSamp)
                   
                    losses: torch.Tensor = (model.calculateLoss(values))
                    #loss_iti = sum([ iti_impl.loss(a, b) for (a,b) in zip(cocoSamp,values)])
                    #losses += loss_iti 
                    
                    if not torch.isnan(losses):
                        losses.backward()
                        #optimizer_iti.step()
                        optimizer.step()
                    optimizer.zero_grad()
                    mb.desc = str(losses.item())
                    l+=losses.item()/len(dataset_train)
                    epochs.desc = str(l)
                    losses = 0

                    model.eval()

                    cocoSamp_ = cocoSamp[0]
                    del cocoSamp
                    cocoSamp :Sample = cocoSamp_
                    detections = model.forward(cocoSamp, dataset=dataset)
                    workImage = cocoSamp.clone()
                    workImage = cocoSamp.detection.onImage(
                                workImage, colors=[(255, 0, 0)])
                    detections=detections.filter(0.05)

                    workImage = detections.NMS_Pytorch().onImage(workImage, colors=[(128, 128, 255)])

                    k=Sample.show(workImage,False, "Training of "+detector)
                    if k == 27:
                        exit()
                if k == 27:
                    break
                torch.save(model.state_dict(),"a2e/"+detector+"_"+name+".fine.pth")
                torch.save(iti_impl.state_dict(),"a2e/"+iti+"_"+name+".fine.pth")
            #torch.save(model.state_dict(),"a2e/"+detector+"_"+name+".fine.pth")
            #torch.save(iti_impl.state_dict(),"a2e/"+iti+"_"+name+".fine.pth")



        with torch.no_grad():
            for sample in tqdm(dataset_eval, leave=False, desc="Evaluating mAP"):
                sample = sample.scale(Size(640,640)).to(device)
                ground_truths.append(sample.detection)
                iti_impl=iti_impl.to(device)
                model=model.to(device)
                detected = model(iti_impl(sample))
                detections_.append(detected)
        def filter_(classIdx):
            return classIdx <=5
        mAP = MultiImageAveragePrecision(ground_truths, detections_)
        del model

        precisions = [AveragePrecision(x,y).precision(0.01) for (x,y) in zip(ground_truths,detections_)]
        torch.cuda.empty_cache()

        precision = sum(precisions) / len(precisions)
        #mAP.filter = filter_
        #mAP=mAP.mAP(0.01)

        addCSV(name,iti,detector,precision)
    #TODO: Only PST900
    break


            

