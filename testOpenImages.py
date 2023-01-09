import fiftyone as fo
import fiftyone.zoo as foz
from interface.datasets import Sample
from interface.datasets.Batch import Batch
from interface.datasets.OpenImages import OpenImagesDetection
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

# Download and load the validation split of Open Images V6
dataset = foz.load_zoo_dataset("open-images-v6", split="validation",max_samples=100,
    seed=51,
    shuffle=False, label_type="detection", classes=["Car"]
    )
#print(dataset.get_classes("detection"))
print(OpenImagesDetection.classesList())
#dataset = foz.load_zoo_dataset("coco-2017", split="validation",max_samples=1,
#    seed=51,
#    shuffle=False,)

#print(foz.datasets.list_zoo_datasets())
dataset = foz.load_zoo_dataset("open-images-v6", split="validation",max_samples=100,
    seed=51,
    shuffle=False, label_type="detection", classes=["Car"], dataset_name="openimagescar"
    )
data = OpenImagesDetection(dataset)

for sample in data:
    sample :fo.Sample
    #print(sample.to_dict())

    #samp:Sample = Sample.fromFiftyOne(sample)
    #show(sample.getRGB(), True)
    #print(samp)
    exit(0)