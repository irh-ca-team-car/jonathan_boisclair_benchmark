import pandas as pd
import random
import torch

try:
    df = pd.read_csv("A3.csv")

    t = torch.tensor(df.values)
    t = t[:,1:]

    mx = t.max(1).values
    potential = t/ mx.view(-1,1).repeat(1,t.shape[1])
    
    SP = potential.mean(0)
    print("SP",SP)
    mIOU = t.mean(0)
    print("mIOU",mIOU)

    df = pd.read_csv("A3_Vit.csv")

    print("vit",df["potential"].mean())

    df = pd.read_csv("A3_Valexnet.csv")

    print("alexnet",df["potential"].mean())
    print("max",df["max"].mean())
    print("alexnet mIOU",df["alexnet"].mean())
except:
    pass
