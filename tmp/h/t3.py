import torch

import t2
import sys
sys.path.append("../")

mdl = torch.load("../file.pt")
print(mdl)