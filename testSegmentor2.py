from interface.segmentation.Segmenter import Segmenter
from interface.datasets.Sample import Sample
import torch
from interface.datasets.Batch import Batch
from interface.datasets import DetectionDataset
from tqdm import tqdm

from interface.transforms.Scale import ScaleTransform
dataset = DetectionDataset.named("voc-2007")

device = "cuda:0"
master_model = Segmenter.named("unet+mit_b0").to(device)
scale = ScaleTransform(224,224)

for name,model_ctr in [("unet+mit_b0", master_model)]:
    model = model_ctr.to(device) #Segmenter.named("deeplabv3_resnet50")

    with torch.no_grad():
        batch=Batch.of(dataset,1)
        for cocoSamp in tqdm(batch):
            cocoSamp = scale(cocoSamp)
            prediction =[f.filter(0.8) for f in model.forward(cocoSamp)]
            Sample.show(prediction[0].onImage(cocoSamp[0]), wait=False, name=name)
            del prediction
            del cocoSamp