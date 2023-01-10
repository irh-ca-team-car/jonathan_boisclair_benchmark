from interface.detectors import Detector, Detection, Box2d
from interface.datasets import Sample, Size
from interface.datasets.A2 import A2Detection
from interface.datasets.Coco import CocoDetection
from interface.datasets.Batch import Batch
import interface
import torch
import time
import cv2
import torchvision

def show(t: torch.Tensor,wait: bool = False):
    if len(t.shape) ==3:
        t=t.unsqueeze(0)
    t = torch.nn.functional.interpolate(t, scale_factor=(1.0,1.0))
    if len(t.shape) ==4:
        t=t[0]
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
        cv2.waitKey(1)

#dataset = CitiscapesDetection(suffix="8bit.png")
dataset = A2Detection("/home/boiscljo/git/pytorch_ros/src/distributed/data/fusiondata/all.csv")
from tqdm import tqdm
models = [(name,det()) for (name,det) in Detector.getAllRegisteredDetectors().items()]
models=[models[-1]]
print([name for (name,det) in models])
for i,(name,det) in enumerate(models):
    model :Detector = det.adaptTo(A2Detection).to("cuda:0")
    model.train()
    optimizer = torch.optim.Adamax(model.parameters())
    losses = 0
    batch=Batch.of(dataset,2)

    for cocoSamp in tqdm(batch):
        cocoSamp = [c.scale(Size(512,400)) for c in cocoSamp]
        losses +=(model.calculateLoss(cocoSamp))

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses=0

        model.eval()

        cocoSamp_=cocoSamp[-1]
        del cocoSamp
        cocoSamp=cocoSamp_
        detections = model.forward(cocoSamp,dataset= A2Detection)
        workImage = cocoSamp.clone()
        workImage = cocoSamp.detection.onImage(workImage, colors=[(255,0,0)])
        workImage = detections.filter(0.5).onImage(workImage)
        show(workImage, False)
        #show(cocoSamp.detection.onImage(cocoSamp), False)
        model.train()
        cv2.waitKey(33)
        #del model
    pass
