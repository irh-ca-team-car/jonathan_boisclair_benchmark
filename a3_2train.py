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

#coco = pycocotools.coco.COCO(annFile)
#coco.download("interface/datasets/coco/imgs", coco.getImgIds(catIds=[3]))

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
preScale = ScaleTransform(640, 640)
randomCrop = RandomCropAspectTransform(400,400,0.2,True)
transform2 = ScaleTransform(480, 352)
rotation = RandomRotateTransform([0,1,2,3,4,5,6,7,8,9,10,90,180,270,359,358,357,356,355,354,353,352,351,350])
autoContrast = AutoContrast()
transforms = [autoContrast,device,preScale,rotation,randomCrop,preScale]
transforms = [autoContrast,device,rotation,randomCrop,preScale]
for b in range(10):
    for dname,dataset in datasets:
        from tqdm import tqdm
        dataset = dataset.withMax(1000)
        models : List[Tuple[str,Detector]] = [
            ("ssd",2,Detector.named("ssd")),
            ("retinanet_resnet50_fpn_v2",1,Detector.named("retinanet_resnet50_fpn_v2")),
            ("ssd_lite",2,Detector.named("ssd_lite")),
            ("yolov5n",4,Detector.named("yolov5n")),
            ("yolov5s",4,Detector.named("yolov5s")),
            ("yolov5m",2,Detector.named("yolov5m")),
        ]
        
        nets = []
        for (mname,bsize, det) in models:
            for itiName,factor in [("Identity",1.5),
            #("VCAE6",1),
            ]:
                iti = ITI.named(itiName)().to(device)
                iti.name = itiName

                if os.path.exists("iti_"+itiName+".pth"):
                    try:
                        iti.load_state_dict(torch.load("iti_"+itiName+".pth", map_location=device), strict=False)
                    except:
                        pass
                    
                model: Detector = det.adaptTo(dataset).to("cpu")
                save_name = "train_"+itiName+"_"+dname+"_"+mname+".pth"
                tmpModule = torch.nn.ModuleList([model,iti])
                if os.path.exists(save_name):
                    try:
                        tmpModule.load_state_dict(torch.load(save_name, map_location=device), strict=False)
                    except:
                        raise Exception(save_name+" does not load")
                #if "retinanet_resnet50_fpn_v2" == mname and itiName == "VCAE6": continue
                nets.append((mname,int(bsize*factor),iti,model,save_name))

        

        for i, (mname,bsize,iti, det,model_path) in enumerate(nets):
            iti:ITI = iti
            det: Detector = det.to(device)
            det.train()
            tmpModule = torch.nn.ModuleList([det,iti])
            optimizer = torch.optim.Adamax(det.parameters())
            if "yolo" in mname:
                optimizer =smart_optimizer(tmpModule)

            t = tqdm(Batch.of(dataset,bsize), leave=True, desc = dname+":"+iti.name+":"+mname)
            for b, samp in enumerate(t):
                optimizer.zero_grad()
                with torch.no_grad():
                    cocoSamp = interface.transforms.apply(samp, transforms)
                    values=iti.forward(cocoSamp)
                losses: torch.Tensor = (det.calculateLoss(values))
                #if torch.isnan(losses):
                #    losses=0
                #if isinstance(cocoSamp,Sample):
                #    loss_iti = iti.loss(cocoSamp,values) 
                #else:
                #    loss_iti = sum([ iti.loss(a, b) for (a,b) in zip(cocoSamp,values)])
                #if not torch.isnan(loss_iti):
                #    losses += (loss_iti*0.1)
                t.desc = dname+":"+iti.name+":"+mname +" "+str(float(losses))

                if not torch.isnan(losses) :
                    losses.backward()
                    optimizer.step()
                optimizer.zero_grad()
                losses = 0

                model.eval()

                if isinstance(cocoSamp,List):
                    cocoSamp_ = cocoSamp[0]
                    values:Sample = values[0]
                    del cocoSamp
                    cocoSamp :Sample = cocoSamp_
                detections = model.forward(cocoSamp, dataset=dataset)
                workImage :Sample= values.clone()
                workImage: torch.Tensor = cocoSamp.detection.onImage(
                    workImage, colors=[(255, 0, 0)])
                #workImage = detections.filter(0.1).onImage(workImage)
                avg = sum([x.cf for x in detections.boxes2d])/len(detections.boxes2d)
                avg = (avg+max([x.cf for x in detections.boxes2d]))/2
                #if avg < 0.3:
                #    avg = 0.3
                tqdm.write(str(avg))

                detections=detections.filter(avg)

                workImage = detections.NMS_Pytorch().onImage(workImage, colors=[(128, 128, 255)])
                #workImage = detections.filter(0.90).onImage(workImage)
                # for b in detections.boxes2d:
                #    print(b)
                if show(workImage, False) >=0:
                    break
                model.train()
                del cocoSamp
                del values
                del detections
                del workImage
            tqdm.write("Writing weights to "+model_path)
            torch.save(tmpModule.state_dict(),model_path)
            tqdm.write("Weights written to "+model_path)
            det.to("cpu")
            del tmpModule
            del det
            del iti
            del optimizer
            import time

            
            break

            time.sleep(2)
            torch.cuda.empty_cache()
        break


                
            


    