import fiftyone
import fiftyone.zoo
from interface.datasets import Sample
from interface.impl.YoloV7 import YoloV7Detector
from interface.detectors import Detector
import torch
import cv2
# Detector.register("yolov7x",YoloV7DetectorInitiator('yolov7x'))
# Detector.register("yolov7",YoloV7DetectorInitiator('yolov7'))
# Detector.register("yolov7-d6",YoloV7DetectorInitiator('yolov7-d6'))
# Detector.register("yolov7-w6",YoloV7DetectorInitiator('yolov7-w6'))
# Detector.register("yolov7-tiny",YoloV7DetectorInitiator('yolov7-tiny'))
model = YoloV7Detector("yolov3").to("cuda:0")
#model = YoloV7Detector("yolov3").to("cuda:0")
#model = Detector.named("yolov5x").to("cuda:0")
sample = Sample.Example()

#detections = model.forward(sample)

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

optimizer = smart_optimizer(model)

while(True):
    if True:# dataset.__class__.getName() != "MS-COCO":
        sample = sample.to("cuda:0")
        optimizer.zero_grad()
        model.train()
        losses: torch.Tensor = (model.calculateLoss([sample]))

        if not torch.isnan(losses):
            print(losses)
            losses.backward()
            optimizer.step()
        optimizer.zero_grad()
        model.eval()

        detections = model.forward(sample)
        workImage = sample.clone()
     
        detections=detections.filter(0.3)

        workImage = detections.NMS_Pytorch().onImage(workImage, colors=[(128, 128, 255)])
        Sample.show(workImage)
        model.train()