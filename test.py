from interface.detectors import Detector, Detection, Box2d
from interface.detectors import Sample
import interface
import torch
import time
import cv2


def show(t: torch.Tensor,wait: bool = False):
    t = torch.nn.functional.interpolate(t, scale_factor=1)
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


sample  = Sample()
det = Detection()
box2d = Box2d()
box2d.x=150
box2d.y=340
box2d.w=130
box2d.h=487-340
box2d.c=3
det.boxes2d.append(box2d)
sample.setTarget(det)

for i,(name,det) in enumerate(Detector.getAllRegisteredDetectors().items()):
    print(name,det)
    model :Detector = det().to("cuda:0")

    optimizer = torch.optim.Adam(model.parameters())
   
    for x in range(50):
        model.train()
        losses=(model.calculateLoss(sample))
        optimizer.zero_grad()
        losses.backward()
        print(name,x,losses.item())
        optimizer.step()

        model.eval()
        detections = model.forward(sample)
        show(detections.filter(0.5).onImage(sample), False)

    cv2.waitKey(1000)
    #detections = model.forward([sample,sample])
    #print([x.filter(0.5) for x in detections])
    del model