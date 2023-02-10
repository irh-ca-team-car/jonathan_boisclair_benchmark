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


datasets : List[Tuple[str,DetectionDataset]] = [
    #("A1_UQTR_REGULAR",A1Detection("data/attention-data/UQTRR/full.txt")),
    #("A2",A2Detection("/home/boiscljo/git/pytorch_ros/src/distributed/data/fusiondata/all.csv")),
    #("FLIR_CONVERTED",A2Detection("data/FLIR_CONVERTED/all.csv")),
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
    ("CitiscapesDetection_8bit",CitiscapesDetection(mode="train", suffix="8bit.png")),
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
itiName = "Identity"
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

preScale = ScaleTransform(640, 640)
randomCrop = RandomCropAspectTransform(400,400,0.2,True)
transform2 = ScaleTransform(480, 352)
rotation = RandomRotateTransform([0,1,2,3,4,5,6,7,8,9,10,90,180,270,359,358,357,356,355,354,353,352,351,350])
autoContrast = AutoContrast()
transforms = [autoContrast,device,preScale,rotation,randomCrop,preScale]
transforms = [autoContrast,device,rotation,randomCrop,preScale]
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
for dname,dataset in datasets:
    break
    from tqdm import tqdm
    #validation loop
    for b, cocoSamp in enumerate(tqdm(dataset)):

        tv = cocoSamp.detection.toTorchVisionTarget()
        boxes = tv["boxes"]
        degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
        if degenerate_boxes.any():
            print(b)
        pass

for dname,dataset in datasets:
    from tqdm import tqdm
    # models = [(name, det())
    #      for (name, det) in Detector.getAllRegisteredDetectors().items()]
    #models = [models[-1]]
    #models = [("EfficientDetector_d0", Detector.named("EfficientDetector_d0"))]
    models : List[Tuple[str,Detector]] = [("retinanet_resnet50_fpn_v2",Detector.named("retinanet_resnet50_fpn_v2"))]
    models : List[Tuple[str,Detector]] = [("yolov5n",Detector.named("yolov5n")),("yolov5s",Detector.named("yolov5s")),("yolov5m",Detector.named("yolov5m")),("yolov5x",Detector.named("yolov5x")),("retinanet_resnet50_fpn_v2",Detector.named("retinanet_resnet50_fpn_v2"))]
    print([name for (name, det) in models])

    for i, (name, det) in enumerate(models):
        model: Detector = det.adaptTo(dataset).to(device)
        
        
        model.train()
        tmpModule = torch.nn.ModuleList([model,iti])

        optimizer = torch.optim.Adamax(tmpModule.parameters())

        print("Parameters",sum(p.data.nelement() for p in tmpModule.parameters()))
        if "yolo" in name:
            optimizer =smart_optimizer(tmpModule)
        losses = 0
        batch = Batch.of(dataset, 2)

        save_name = itiName+"_"+dname+"_"+name+".pth"
        if os.path.exists(save_name):
            try:
                tmpModule.load_state_dict(torch.load(save_name, map_location=device), strict=False)
            except:
                pass
        t = tqdm(batch, leave=True)
        for b, cocoSamp in enumerate(t):
            #cocoSamp = [c.scale(Size(512, 416)).to(device) for c in cocoSamp]
            #cocoSamp = [c.scale(Size(752, 480)).to(device) for c in cocoSamp]
            #cocoSamp = [c.scale(Size(480, 352)).to(device) for c in cocoSamp]
            cocoSamp = interface.transforms.apply(cocoSamp, transforms)
            
            if True:# dataset.__class__.getName() != "MS-COCO":
                values=iti.forward(cocoSamp)
                losses: torch.Tensor = (model.calculateLoss(values))
                loss_iti = sum([ iti.loss(a, b) for (a,b) in zip(cocoSamp,values)])
                losses += loss_iti 
                optimizer.zero_grad()
                t.desc = name +" "+str(losses.item())

                if not torch.isnan(losses):
                    losses.backward()
                    #tqdm.write(str(losses.item()))
                    optimizer.step()
                optimizer.zero_grad()
                losses = 0
            

            model.eval()

            cocoSamp_ = cocoSamp[0]
            values = values[0]
            del cocoSamp
            cocoSamp :Sample = cocoSamp_
            detections = model.forward(cocoSamp, dataset=dataset)
            workImage = values.clone()
            workImage = cocoSamp.detection.onImage(
                workImage, colors=[(255, 0, 0)])
            #workImage = detections.filter(0.1).onImage(workImage)
            detections=detections.filter(0.3)

            workImage = detections.NMS_Pytorch().onImage(workImage, colors=[(128, 128, 255)])
            #workImage = detections.filter(0.90).onImage(workImage)
            # for b in detections.boxes2d:
            #    print(b)
            if show(workImage, False) >=0:
                break
            #show(cocoSamp.detection.onImage(cocoSamp), False)
            model.train()
            cv2.waitKey(1)
            #del model
        optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        torch.save(tmpModule.state_dict(),save_name)
        pass
