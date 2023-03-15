import fiftyone
import fiftyone.zoo
from interface.datasets import Sample
from interface.datasets.detection import A2Detection, CocoDetection
from interface.impl.A2Det import A2Det
from interface.impl.YoloV8 import YoloV8Detector
from interface.detectors import Detector
import torch
import cv2

def validate_state_dicts(model_state_dict_1, model_state_dict_2):
    if len(model_state_dict_1) != len(model_state_dict_2):
        print(
            f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}"
        )
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_1.items()
        }

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_2.items()
        }

    for ((k_1, v_1), (k_2, v_2)) in zip(
        model_state_dict_1.items(), model_state_dict_2.items()
    ):
        if k_1 != k_2:
            print(f"Key mismatch: {k_1} vs {k_2}")
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2):
            print(f"Tensor mismatch: {k_1} vs {k_2}")

        
device = "cuda:0"
# Detector.register("yolov7x",YoloV7DetectorInitiator('yolov7x'))
# Detector.register("yolov7",YoloV7DetectorInitiator('yolov7'))
# Detector.register("yolov7-d6",YoloV7DetectorInitiator('yolov7-d6'))
# Detector.register("yolov7-w6",YoloV7DetectorInitiator('yolov7-w6'))
# Detector.register("yolov7-tiny",YoloV7DetectorInitiator('yolov7-tiny'))
model = A2Det(sep=3,mdl="src/distributed/research-config/proof-of-concept-split-ssd-attention/models/model-ssd-alexnet.cfg").adaptTo(A2Detection()).to(device)


#model = YoloV7Detector("yolov3").to(device)
#model = Detector.named("yolov5x").to(device)
sample = Sample.Example()
#for x in sample.detection.boxes2d:
#    x.c = CocoDetection().getId("car")

import os
if not os.path.exists("w"): os.mkdir("w")

model.train()
try:
    d=torch.load("w/"+model.model_name+".pth", map_location=device)
    model.load_state_dict(d)
    pass
except BaseException as e:
    print("Could not load", e)
    pass
losses: torch.Tensor = (model.calculateLoss([sample]))


param_names= set()
for n,param in model.model.named_parameters():
    param_names.add(n)

d=model.state_dict()
weight_names= set(d.keys())

missing = weight_names - param_names

print(missing)

#print(model.model)
print(losses)

#print(d.keys())

torch.save(d, "w/"+model.model_name+".pth")

model = A2Det(sep=3,mdl="src/distributed/research-config/proof-of-concept-split-ssd-attention/models/model-ssd-alexnet.cfg").adaptTo(A2Detection()).to(device)
model.train()
try:
    d=torch.load("w/"+model.model_name+".pth", map_location=device)
    model.load_state_dict(d)
    pass
except BaseException as e:
    print("Could not load", e)
    pass
losses: torch.Tensor = (model.calculateLoss([sample]))
print(losses)


sd = model.state_dict()

validate_state_dicts(d,sd)
print(d.keys())
exit(0)
optimizer=YoloV8Detector.optimizer(model)



det = model.forward([sample])
det = model.forward(sample)

workImage = det.filter(0.3).onImage(sample, colors=[(128, 128, 255)])
Sample.show(workImage,False)

#optimizer = model.trainer.build_optimizer(model.model.model)
sample = sample.to(device)

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