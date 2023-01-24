import torch

model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True, trust_repo=True)

img = torch.randn(1,3,640,640)
det_out, da_seg_out,ll_seg_out = model(img)

print(det_out,da_seg_out,ll_seg_out)