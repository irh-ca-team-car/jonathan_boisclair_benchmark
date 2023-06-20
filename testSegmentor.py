from interface.segmentation.Segmenter import Segmenter
from interface.datasets.Sample import Sample
import torch
from interface.transforms.Scale import ScaleTransform
scale = ScaleTransform(224,224)

device = "cuda:0"
img = Sample.Example().to(device)
master_model = Segmenter.named("deeplabv3_resnet50").to(device)
img.segmentation = master_model.forward(img)
master_model = Segmenter.named("fcn_resnet101").to(device)

optim = torch.optim.Adamax(master_model.parameters(),lr=2e-6)
for b in range(5):
    optim.zero_grad()
    loss = (master_model.calculateLoss([img]))
    loss.backward()
    prediction =[f.filter(0.8) for f in master_model.forward([img])]
    print(loss)
    optim.step()
    Sample.show(prediction[0].onImage(img), wait=False, name="pre-train")

    optim.zero_grad()
img.segmentation = master_model.forward(img)

model_1 = Segmenter.named("unet+mit_b0").to(device)
def mdl():
    return model_1

#for name,model_ctr in Segmenter.registered_Segmenters.items():
for i in range(4):
    for name,model_ctr in [("unet+mit_b0", mdl)]:
        model = model_ctr().to(device) #Segmenter.named("deeplabv3_resnet50")
        model.freeze_backbone()
        optim = torch.optim.Adamax(model.parameters(),lr=2e-3)
        for b in range(500):
            optim.zero_grad()
            loss = (model.calculateLoss([img]))
            loss.backward()
            prediction =[f.filter(0.8) for f in model.forward([img])]
            print(loss)
            optim.step()
            Sample.show(prediction[0].onImage(img), wait=False, name=name)

            optim.zero_grad()
