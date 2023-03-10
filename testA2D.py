import fiftyone
import fiftyone.zoo
from interface.datasets import Sample
from interface.datasets.detection import A2Detection, CocoDetection
from interface.impl.A2Det import A2Det
from interface.impl.YoloV8 import YoloV8Detector
from interface.detectors import Detector
import torch
import cv2
# Detector.register("yolov7x",YoloV7DetectorInitiator('yolov7x'))
# Detector.register("yolov7",YoloV7DetectorInitiator('yolov7'))
# Detector.register("yolov7-d6",YoloV7DetectorInitiator('yolov7-d6'))
# Detector.register("yolov7-w6",YoloV7DetectorInitiator('yolov7-w6'))
# Detector.register("yolov7-tiny",YoloV7DetectorInitiator('yolov7-tiny'))
model = A2Det(sep=0).adaptTo(A2Detection()).to("cuda:0")


#model = YoloV7Detector("yolov3").to("cuda:0")
#model = Detector.named("yolov5x").to("cuda:0")
sample = Sample.Example()
#for x in sample.detection.boxes2d:
#    x.c = CocoDetection().getId("car")

import os
if not os.path.exists("w"): os.mkdir("w")

try:
    d=torch.load("w/"+model.model_name+".pth", map_location="cuda:0")
    #model.load_state_dict(d,False)
    model.load_state_dict(d)
    pass
except BaseException as e:
    print("Could not load", e)
    pass
optimizer=YoloV8Detector.optimizer(model)
model.train()
losses: torch.Tensor = (model.calculateLoss([sample]))
print(losses)

det = model.forward([sample])
det = model.forward(sample)

workImage = det.filter(0.3).onImage(sample, colors=[(128, 128, 255)])
Sample.show(workImage,False)

#optimizer = model.trainer.build_optimizer(model.model.model)
sample = sample.to("cuda:0")

epoch = 0
while(True):
    epoch = epoch +1
    if True:# dataset.__class__.getName() != "MS-COCO":
        
        optimizer.zero_grad()
        model.train()
        
        losses: torch.Tensor = (model.calculateLoss([sample]))

        print(losses,epoch)
        if not torch.isnan(losses):
            losses.backward()
            del losses
            optimizer.step()
        optimizer.zero_grad()
        model.eval()

        detections = model.forward(sample)
        workImage = sample.clone()
     
        detections=detections

        workImage = detections.onImage(workImage, colors=[(128, 128, 255)])
        del detections
        k=Sample.show(workImage)
        if(k == 115):
            with torch.inference_mode():
                torch.save(model.state_dict(),"w/"+model.model_name+".pth")
                break
        if(k==113):
            exit(0)
        del workImage
        model.train()