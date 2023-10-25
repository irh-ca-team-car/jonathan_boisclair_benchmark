import pandas as pd
import random
import torch

try:
    for p in ["A3.csv","A3_snow.csv","A3_water.csv"]:
        df = pd.read_csv(p)
        cols = list(df)[1:]
        cols.sort()
        df = df[cols]
        print(df.columns)
        t = torch.tensor(df.values)
        t = t[:,:]

        mx = t.max(1).values
        potential = t/ mx.view(-1,1).repeat(1,t.shape[1])
        
        SP = potential.mean(0)
        print("SP",p,SP)
        mIOU = t.mean(0)
        print("mIOU",p,mIOU)

    df = pd.read_csv("A3_Vit.csv")

    print("vit",df["potential"].mean())

    df = pd.read_csv("A3_Valexnet.csv")

    print("alexnet",df["potential"].mean())
    print("max",df["max"].mean())
    print("alexnet mIOU",df["alexnet"].mean())
except:
    pass
