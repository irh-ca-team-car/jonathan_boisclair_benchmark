from interface.detectors import Detector, Detection, Box2d
from interface.datasets import Sample, Size
from interface.datasets.Citiscapes import CitiscapesDetection
from interface.datasets.Coco import CocoDetection
from interface.datasets.Batch import Batch
from interface.datasets.OpenImages import OpenImagesDetection
from interface.impl import EfficientDetector
import interface
import torch
import time
import cv2
import torchvision
import torchvision.transforms as transforms
import pycocotools.coco
import fiftyone.zoo as foz

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
        cv2.waitKey(1)


datasets = [
    OpenImagesDetection(dataset=foz.load_zoo_dataset("open-images-v6",
                                                                         split="validation",
                                                                         max_samples=100,
                                                                         seed=51,
                                                                         shuffle=False,
                                                                         label_type="detection",
                                                                         classes=[
                                                                             "Car"],
                                                                         dataset_name="openimagescar"
                                                                         )),
    CitiscapesDetection(mode="train", suffix="8bit.png"),
    #CitiscapesDetection(mode="train", suffix="0.005.png"),
    #CitiscapesDetection(mode="train", suffix="0.01.png"),
    #CitiscapesDetection(mode="train", suffix="0.02.png"),
    CitiscapesDetection(mode="val", suffix="8bit.png"),
    #CitiscapesDetection(mode="val", suffix="0.005.png"),
    #CitiscapesDetection(mode="val", suffix="0.01.png"),
    #CitiscapesDetection(mode="val", suffix="0.02.png"),
    CocoDetection("interface/datasets/coco/imgs", annFile)
]
#dataset = CitiscapesDetection(suffix="8bit.png")
#dataset = CitiscapesDetection(suffix="0.02.png")

for dataset in datasets:
    from tqdm import tqdm
    # models = [(name, det())
    #      for (name, det) in Detector.getAllRegisteredDetectors().items()]
    #models = [models[-1]]
    #models = [("EfficientDetector_d0", Detector.named("EfficientDetector_d0"))]
    models = [("retinanet_resnet50_fpn_v2",Detector.named("retinanet_resnet50_fpn_v2"))]
    print([name for (name, det) in models])

    for i, (name, det) in enumerate(models):
        model: Detector = det.adaptTo(dataset.__class__).to("cuda:0")
        model.train()
        optimizer = torch.optim.Adamax(model.parameters())
        losses = 0
        batch = Batch.of(dataset, 1)

        for b, cocoSamp in enumerate(tqdm(batch)):
            cocoSamp = [c.scale(Size(512, 418)) for c in cocoSamp]

            # if True:# dataset.__class__.getName() != "MS-COCO":
            for b_ in range(5):
                
                losses = (model.calculateLoss(cocoSamp))

                optimizer.zero_grad()
                losses.backward()
                if b_ == 4:
                    tqdm.write(str(losses.item()))
                optimizer.step()
                optimizer.zero_grad()
                losses = 0
            

            model.eval()

            cocoSamp_ = cocoSamp[-1]
            del cocoSamp
            cocoSamp = cocoSamp_
            detections = model.forward(cocoSamp, dataset=dataset.__class__)
            workImage = cocoSamp.clone()
            workImage = cocoSamp.detection.onImage(
                workImage, colors=[(255, 0, 0)])
            #workImage = detections.filter(0.1).onImage(workImage)
            workImage = detections.filter(0.3).onImage(workImage)
            #workImage = detections.filter(0.90).onImage(workImage)
            # for b in detections.boxes2d:
            #    print(b)
            show(workImage, False)
            #show(cocoSamp.detection.onImage(cocoSamp), False)
            model.train()
            cv2.waitKey(100)
            #del model
        pass
