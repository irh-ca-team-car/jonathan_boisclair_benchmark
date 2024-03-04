import pandas as pd
import random
import torch

try:
    for p in ["A3.csv","A3_snow.csv","A3_water.csv","A3_wr.csv","A3_snow_wr.csv","A3_water_wr.csv"]:
        df = pd.read_csv(p)
        cols = list(df)[1:]
        cols.sort()
        df = df[cols]
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

    for p in ["","_water","_snow"]:
        df = pd.read_csv("A3_V"+p+"alexnet.csv")

        print(p,"alexnet",df["potential"].mean())
        print(p,"max",df["max"].mean())
        print(p,"alexnet mIOU",df["alexnet"].mean())
except:
    pass
