from typing import List, Tuple
from interface.datasets.detection.A2 import A2Detection
from interface.datasets.detection.A2W import A2W
from interface.detectors import Detector
from interface.datasets import DetectionDataset,Detection, Sample, Size
from interface.datasets.Batch import Batch
from interface.ITI import ITI
from interface.metrics.Metrics import AveragePrecision, MultiImageAveragePrecision
from interface.transforms import apply, FLIR_FIX
from interface.transforms.RandomCut import RandomCropAspectTransform
from interface.transforms.RandomRotate import RandomRotateTransform
from interface.transforms.Scale import ScaleTransform
import torch
from tqdm import tqdm

from interface.transforms.TorchVisionFunctions import AutoContrast
configs = [
    #("VCAE6","yolov5n"),
    #("Identity","yolov5n"),
    #("DenseFuse","yolov5n"),
    ("VCAE6","yolov7-tiny"),
    ("Identity","yolov7-tiny"),
    ("DenseFuse","yolov7-tiny"),
    #("Identity","ssd"),
    #("Identity","fasterrcnn_resnet50_fpn"),
]

datasets : List[Tuple[str,DetectionDataset]] = [
    ("A2",A2W(A2Detection("/home/boiscljo/git/pytorch_ros/src/distributed/data/fusiondata/all.csv"))),
    ("FLIR_CONVERTED",A2W(A2Detection("data/FLIR_CONVERTED/all.csv")))]
datasets_train : List[Tuple[str,DetectionDataset]] = [
    ("A2",A2W(A2Detection("/home/boiscljo/git/pytorch_ros/src/distributed/data/fusiondata/all.csv"))),
    ("FLIR_CONVERTED",A2W(A2Detection("data/FLIR_CONVERTED/all.csv")))]
datasets_eval : List[Tuple[str,DetectionDataset]] = [
    ("A2",A2W(A2Detection("/home/boiscljo/git/pytorch_ros/src/distributed/data/fusiondata/all.csv"))),
    ("FLIR_CONVERTED",A2W(A2Detection("data/FLIR_CONVERTED/all.csv")))]
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
randomCrop = RandomCropAspectTransform(400,400,0.2,True)
transform2 = ScaleTransform(480, 352)
rotation = RandomRotateTransform([0,1,2,3,4,5,6,7,8,9,10,359,358,357,356,355,354,353,352,351,350])
autoContrast = AutoContrast()
transforms = [autoContrast,rotation,randomCrop,preScale]

def smart_optimizer(model, name='Adam', lr=2.4e-5, momentum=0.9, decay=1e-6):
    import torch.nn as nn
    # YOLOv5 3-param group optimizer: 0) weights with decay, 1) weights no decay, 2) biases no decay
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == 'bias':  # bias (no decay)
                g[2].append(p)
            elif p_name == 'weight' and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # weight (with decay)

    if name == 'Adam':
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == 'RMSProp':
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f'Optimizer {name} not implemented.')

    optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
 
    return optimizer

for (name,dataset),(_,dataset_train),(_,dataset_eval) in zip(datasets,datasets_train,datasets_eval):
    for iti,detector in configs:
        print(name,iti,detector)
        detections_ = []
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
            optimizer = smart_optimizer(model)
            epochs = tqdm(range(1000), leave=False)
            for b in epochs:
                dts = datasets[1][1].withMax(180)
                bts = Batch.of(dts,20)

                inner = tqdm(bts, leave=False)
                for cocoSamp in inner:
                    model.train()
                    cocoSamp=apply(cocoSamp,[FLIR_FIX,"cuda:0",*transforms])
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
            dataset_eval.images = dataset.images[300:1000]
        else:
            dataset_train.images = dataset.images[0:int(len(dataset.images)/2)]
            dataset_eval.images = dataset.images[int(len(dataset.images)/2):]

        if True:
            model.train()
            optimizer = smart_optimizer(model)
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
                    detections=detections.filter(0.05)

                    workImage = detections.NMS_Pytorch().onImage(workImage, colors=[(128, 128, 255)])

                    k=Sample.show(workImage,False, "Training of "+detector)
                    if k == 27:
                        exit()
                if k == 27:
                    break



        with torch.no_grad():
            for sample in tqdm(dataset_eval, leave=False, desc="Evaluating mAP"):
                sample = sample.scale(Size(640,640)).to(device)
                ground_truths.append(sample.detection)
                detected = model(iti_impl(sample))
                detections_.append(detected)
        def filter_(classIdx):
            return classIdx <=5
        mAP = MultiImageAveragePrecision(ground_truths, detections_)


        precisions = [AveragePrecision(x,y).precision(0.5) for (x,y) in zip(ground_truths,detections_)]

        precision = sum(precisions) / len(precisions)
        #mAP.filter = filter_
        #mAP=mAP.mAP(0.01)

        addCSV(name,iti,detector,precision)


            

