from dataclasses import dataclass, field
from typing import Dict
from interface.segmentation.Segmenter import Segmenter
from interface.datasets.Sample import Sample
import torch
from interface.datasets.Batch import Batch
from interface.datasets import DetectionDataset
from interface.datasets.detection.CocoFO import CocoFODetection
from interface.metrics.Metrics import mIOU, mIOUAccumulator
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import random
import math
from interface.transforms.Scale import ScaleTransform
import os
#dataset = DetectionDataset.named("voc-2007")

dataset = CocoFODetection(split="train",max_samples=10000, type_=["segmentations"], classes=["car","truck","traffic light","stop sign","bus","person","bicycle","motorcycle"])
dataset.lazy()

if(len(dataset)==0):
    print("reload")
    exit(5)
device = "cuda:0"

import sys

#m_name="unet+tu-tinynet_a"
try:
    os.mkdir("a3_weights")
except:
    pass

@dataclass
class settings():
    processed:Dict = field(default_factory=dict)
    current_epoch:int =10

state = settings()
try:
    state= torch.load("a3_weights/state.chk")
except:
    pass

scale = ScaleTransform(224,224)
need_exit = False

if len(sys.argv) <2:
    print("Error, missing arguments ",__file__,"SEGMENTER")
    exit(2)

try:
    model_name = sys.argv[1]
except:
    print("Error,Invalid argument ",__file__,"SEGMENTER")
    exit(2)

for name in tqdm([model_name],desc="models",leave=False):
        print(name,state.processed[name], "to", state.current_epoch)
        
        if name in state.processed and state.processed[name] >= state.current_epoch:
            print("Already over")
            continue
        if name not in state.processed:
            state.processed[name]=0

        model_ctr = Segmenter.named(name).to(device)
        model = model_ctr.adaptTo(dataset).to(device) #Segmenter.named("deeplabv3_resnet50")
        try:
            model.load_state_dict(torch.load("a3_weights/"+name+".pth", map_location=device), strict=False)
        except:
            pass
        optim: torch.optim.Optimizer = model.optimizer(model,2e-3,0) 
        # Assuming optimizer has two groups.
        lambda_group1 = lambda epoch: (5+(random.random()*2-1)) *10** (-(6+float(epoch) / 1000 - float(epoch) // 1000))
        lambda_group2 = lambda epoch: (5+(random.random()*2-1)) *10** (-(10+float(epoch) / 1000- float(epoch) // 1000))

        for i in tqdm(range(state.processed[name],state.current_epoch),desc="epoch",leave=False):
            if i>=5:
                model.unfreeze_backbone()
            else:
                model.freeze_backbone()

            b_size = min(math.ceil(float(i+1)/25.0),4)
            batch=Batch.of(dataset,b_size)
            
            iter=0
            mIou = None
            t=tqdm(batch,desc="batch",leave=False)
            acc = mIOUAccumulator(len(dataset.classesList()))

        
            
            
            for cocoSamp in t:
                cocoSamp = scale(cocoSamp)
                optim.param_groups[0]["lr"] = lambda_group1(iter)
                optim.param_groups[1]["lr"] = lambda_group2(iter)
               
                optim.zero_grad()
                loss = (model.calculateLoss(cocoSamp)) / len(cocoSamp)
                loss.backward()
                prediction =[f.filter(0.8) for f in model.forward(cocoSamp)]

                need_exit = Sample.show(cocoSamp[0].segmentation.onImage(cocoSamp[0]), wait=False, name="gt_"+str(0))  == 27
                need_exit = need_exit or Sample.show(prediction[0].onImage(cocoSamp[0]), wait=False, name=name) == 27
                
                for gt,val in zip([x.segmentation for x in cocoSamp],prediction):
                    acc.acculumate(val,gt)

                optim.step()

                mIou = mIOU([x.segmentation for x in cocoSamp],prediction)
                t.desc = "batch "+str(acc.val())+":"+str(mIou.calc())
                optim.zero_grad()

            
                iter+= len(cocoSamp)
                if need_exit:
                    break

            if acc is not None:
                line = str(i)+"\t"+str(acc.val())+"\t"+str(optim.param_groups[0]["lr"])+"\t"+str(optim.param_groups[1]["lr"])+"\n"
                tqdm.write(line)
                with open("a3_weights/"+name+".txt", "a") as myfile:
                    myfile.write(line)
            
            state.processed[name] += 1
            torch.save(state, "a3_weights/state.chk")
            torch.save(model.state_dict(), "a3_weights/"+name+".pth")
            if need_exit:
                exit(0)
                break
        if need_exit:
            exit(0)
            break
exit(0)