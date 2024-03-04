from typing import List
import dynamics
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import pandas as pd

mr = 2750

data: List = []
latex = ""
for speed in [10,20,30,40,50,60,70,80,90,100]:
    velocity: float = speed / 3.6
    
    benchmark: float = dynamics.generate_optimal_profile(velocity, 500, 0, max_regen=mr, Distance_tolerance=1000, Time_step=30)[0].regenerated

    row = []
    print("speed=",speed,"regenerated=",benchmark)
    for epsilon in [0,0.25,0.50,0.75,0.9]:
        print("evaluating for epsilon=",epsilon)
        braking_distance = float('inf') 
        for d in range(0,300,2):
            profiles: List[dynamics.Path] = dynamics.generate_optimal_profile(velocity, d, 0, max_regen=mr, Distance_tolerance=1000, Time_step=30)
            if(len(profiles)>0):
                energy: int = profiles[0].regenerated
                if(energy > benchmark*epsilon):
                    braking_distance: int = d
                    break
        print("epsilon=",epsilon,"speed=",speed, "braking_distance=",braking_distance)
        row.append(braking_distance)
    data.append(row)
    #0km/h    & 0m                                                     & 0m     & 0m     & 0m     & 0m     \\
    latex += str(speed)+"km/h&"+ ("&".join([str(x)+"m" for x in row]))+"\\\\\n"

df = pd.DataFrame(data, columns=["0","25","50","75","90"])
print(df)
print("---")
print(latex)
            

