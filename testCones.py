from typing import List
from interface.detectors import Detector, Detection, Box2d
from interface.datasets import Sample, Size
from interface.datasets.detection.Cones import ConesDetection
from interface.datasets.Batch import Batch
import interface
import torch
import time
import cv2
import torchvision
from interface.transforms import RandomCutTransform, RandomRotateTransform, rotate, AutoContrast, Cat
from interface.transforms import ScaleTransform
import interface.transforms

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

transform2 = ScaleTransform(640,640)
randomCrop = RandomCutTransform(100,100,0.2,True)
transform2 = ScaleTransform(640,640)
autoContrast = AutoContrast()
transforms = [randomCrop, transform2, autoContrast]
#dataset = CitiscapesDetection(suffix="8bit.png")
dataset = ConesDetection("/home/boiscljo/Downloads/cones")
from tqdm import tqdm
models = [("ssd_lite",Detector.named("ssd_lite"))]
models=[models[-1]]
print([name for (name,det) in models])
def rgb2hsv_torch(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)
def rgb2hsv(rgb: List[Sample]):
    tensor = rgb2hsv_torch(Cat().__call__(rgb))
    for i in range(len(rgb)):
        rgb[i].setImage(tensor[i])
    return rgb
for i,(name,det) in enumerate(models):
    model :Detector = det.adaptTo(dataset).to("cuda:0")
    model.train()
    model.load_state_dict(torch.load("cones.pth"))
    optimizer = torch.optim.Adamax(model.parameters())
    losses = 0
    batch=Batch.of(dataset,2)
    for epc in tqdm(range(100)):
        for cocoSamp in tqdm(batch, leave=False):
            cocoSamp = [c.scale(Size(512,400)) for c in cocoSamp]
            cocoSamp = interface.transforms.apply(cocoSamp,transforms)
            cocoInp = interface.transforms.apply(cocoSamp, [rgb2hsv])

            losses +=(model.calculateLoss(cocoSamp))

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses=0

            model.eval()

            cocoSamp_=cocoInp[-1]
            del cocoSamp
            cocoSamp=cocoSamp_
            detections = model.forward(cocoSamp,dataset=ConesDetection)
            workImage = cocoSamp.clone()
            workImage = cocoSamp.detection.onImage(workImage, colors=[(255,0,0)])
            workImage = detections.filter(0.5).onImage(workImage)
            show(workImage, False)
            #show(cocoSamp.detection.onImage(cocoSamp), False)
            model.train()
            cv2.waitKey(33)
            #del model
        pass
    torch.save(model.state_dict(),"cones.pth")