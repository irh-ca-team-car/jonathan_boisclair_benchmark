import fiftyone as fo
import fiftyone.zoo as foz
from interface.datasets import Sample,Size
from interface.datasets.Batch import Batch
from interface.datasets.detection.PST900 import PST900Detection
from interface.transforms import ScaleTransform
import torch
import cv2

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

data = PST900Detection().withSkip(10)

tf = ScaleTransform(Size(800,800))

for sample in data:
    sample :Sample
    sample = tf(sample)
    tensor = sample.segmentation.onImage(sample, alpha=0.6)
    tensor = sample.detection.onImage(tensor, colors=[(128,128,255)])
    #tensor = sample._segmentation.colored()
    sample.show(tensor,True)
    #sample.getLidar().view()