from interface.segmentation.Segmenter import Segmenter
from interface.datasets.Sample import Sample
import torch
from interface.datasets.Batch import Batch
from interface.datasets import DetectionDataset
from interface.metrics.Metrics import mIOU
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import random
import math
from interface.transforms.Scale import ScaleTransform
dataset = DetectionDataset.named("voc-2007")

device = "cuda:0"
#m_name="unet+tu-tinynet_a"

m_name="unet+mit_b1"
model_1 = Segmenter.named(m_name).to(device)

master_model = Segmenter.named("deeplabv3_resnet50").to(device)

scale = ScaleTransform(224,224)

for name,model_ctr in [(m_name, model_1)]:
    model = model_ctr.adaptTo(dataset).to(device) #Segmenter.named("deeplabv3_resnet50")
    try:
        model.load_state_dict(torch.load(name+".pth", map_location=device), strict=False)
    except:
        pass
    optim: torch.optim.Optimizer = model.optimizer(model,2e-3,0) 
    # Assuming optimizer has two groups.
    lambda_group1 = lambda epoch: (5+(random.random()*2-1)) *10** (-(6+float(epoch) / 1000 - float(epoch) // 1000))
    lambda_group2 = lambda epoch: (5+(random.random()*2-1)) *10** (-(10+float(epoch) / 1000- float(epoch) // 1000))

    #scheduler = LambdaLR(optim, lr_lambda=[lambda_group1, lambda_group2])
    for i in range(400):
        if i>=-1:
            model.unfreeze_backbone()
        else:
            model.freeze_backbone()
        b_size = min(math.ceil(float(i+1)/25.0),8)
        batch=Batch.of(dataset,b_size)
        iter=0
        print(len(batch))
        for cocoSamp in tqdm(batch):
            cocoSamp = scale(cocoSamp)
            optim.param_groups[0]["lr"] = lambda_group1(iter)
            optim.param_groups[1]["lr"] = lambda_group2(iter)
            with torch.no_grad():
                segmentations = master_model(cocoSamp)
                for i in range(len(cocoSamp)):
                    cocoSamp[i].segmentation = segmentations[i]
                    #Sample.show(segmentations[i].onImage(cocoSamp[i]), wait=False, name="gt_"+str(i))
            optim.zero_grad()
            loss = (model.calculateLoss(cocoSamp)) / len(cocoSamp)
            loss.backward()
            prediction =[f.filter(0.8) for f in model.forward(cocoSamp)]
            Sample.show(segmentations[0].onImage(cocoSamp[0]), wait=False, name="gt_"+str(0))
            Sample.show(prediction[0].onImage(cocoSamp[0]), wait=False, name=name)
            optim.step()

            mIou = mIOU([x.segmentation for x in cocoSamp],prediction)
            tqdm.write(str(mIou.calc())+" lr="+str(optim.param_groups[0]["lr"])+" lr2="+str(optim.param_groups[1]["lr"]))
            optim.zero_grad()

           
            iter+= len(cocoSamp)
            #scheduler.step()

        torch.save(model.state_dict(), name+".pth")


# for name,model_ctr in [("unet+mit_b0", model_1)]:
#     model = model_ctr.to(device) #Segmenter.named("deeplabv3_resnet50")

#     with torch.no_grad():
#         batch=Batch.of(dataset,1)
#         for cocoSamp in tqdm(batch):
#             cocoSamp = scale(cocoSamp)
#             prediction =[f.filter(0.8) for f in model.forward(cocoSamp)]
#             Sample.show(prediction[0].onImage(cocoSamp[0]), wait=False, name=name)
#             del prediction
#             del cocoSamp