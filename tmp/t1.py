import torch

import h.t2

mdl = h.t2.T2("noodle")
print(mdl)
torch.save(mdl,"file.pt")