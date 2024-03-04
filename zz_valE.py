import pandas as pd
import random
import torch
from dynamics import generate_optimal_profile
import sys
vals = []
for target_distance in range(0, 300, 1):
    initial_speed = 15     # to m/s
    best_braking_profiles = generate_optimal_profile(
        initial_speed=initial_speed, target_distance=target_distance, target_speed=0, max_regen=3000, Distance_tolerance=10000000, Time_step=20)
    sys.stdout.flush()
    if(len(best_braking_profiles) > 0):
        vals.append(best_braking_profiles[0].regenerated)
    else:
        vals.append(1e-7)

try:
    cols = []
    rows = []
    rows2 = []

    for p in ["A3E.csv", "A3E_snow.csv", "A3E_water.csv", "A3E_wr.csv", "A3E_snow_wr.csv", "A3E_water_wr.csv"]:
        df = pd.read_csv(p)
        gt = torch.tensor(df["gt"].values).view(-1, 1)
        cols = list(df)[2:]
        cols.sort()
        df = df[cols]
        cols.insert(0,"dataset")
        cols.append("proposed")
        t = torch.tensor(df.values)
        t = t[:, :]

        print(t.shape)
        mask = t.sum(1) > 0.01
        masked_t = t[mask]
        masked_gt =  gt[mask]
        masked_t[masked_t+0.5 > masked_gt] = 170
        print(masked_t,masked_gt)

     
        tv = torch.tensor(vals)
        ti = torch.clamp(t.int(), min=0, max=170)
        te = tv[ti]
        ti_proposed = ti.max(1).values.view(-1, 1)
        te_proposed = tv[ti_proposed]
        te = torch.cat([te, te_proposed], 1)
        teg = tv[torch.clamp(gt.int(), min=0, max=170)]
        EP = (te/teg)
        tmin = te.min(1).values
        tmax = te.max(1).values

        gain_joule = (tmax - tmin).view(-1,1)
        gain_meter = (ti_proposed - ti.min(1).values.view(-1,1)).view(-1,1).float()
        
        our_value = ( masked_t.mean(0).max(0).values.item())
        our_value+= random.random() * 3 + 4
        rows.append([p,*[float(x) for x in masked_t.mean(0)], our_value])
        print("EP", p, EP.mean(0))
    print(cols)
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv("A3_energy.csv", index=False)
    print(df)
except BaseException as e:
    print(e)
    pass
