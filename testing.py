import torch

tuple_ = torch.load("test.pt")

print(len(tuple_))

t1,t2,t3,t4= tuple_

print(t1)
print(t2)
print(t3)
print(t4)