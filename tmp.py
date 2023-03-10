import torch
a = torch.randn((8072,6))

print(a.shape)

argmax = torch.argmax(a,1) > 2
argmax=argmax.view([*argmax.shape,1]).repeat(1,6)

print(a[argmax].view(-1,6).shape)

