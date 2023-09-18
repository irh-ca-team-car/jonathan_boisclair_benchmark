from interface.detectors import Detector, Detection, Box2d
from interface.datasets import Sample, Size
from interface.datasets import DetectionDataset
from interface.datasets.Batch import Batch
import interface
import torch
import time
import cv2
import torchvision

# import segmentation_models_pytorch as smp

# model = smp.Unet(
#     encoder_name="mit_b2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=27,                      # model output channels (number of classes in your dataset)
# )

# import torch

# img = torch.randn(1,3,512,512)
# print(model(img).grad_fn)

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
dataset = DetectionDataset.named("voc-2007")
from tqdm import tqdm
models =[("model", Detector.named("ssd_lite") )]
print([name for (name,det) in models])
for i,(name,det) in enumerate(models):
    model :Detector = det.adaptTo(dataset).to("cuda:0")
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
        detections = model.forward(cocoSamp,dataset= dataset)
        workImage = cocoSamp.clone()
        workImage = cocoSamp.detection.onImage(workImage, colors=[(255,0,0)])
        workImage = detections.filter(0.5).onImage(workImage)
        show(workImage, False)
        #show(cocoSamp.detection.onImage(cocoSamp), False)
        model.train()
        cv2.waitKey(33)
        #del model
    pass
