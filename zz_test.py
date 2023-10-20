import pandas as pd
import random

C_name = input("C_name")

image_id = 0
value = random.random()
df = None
def append_to_csv(C_name, image_id, value, save=True):
    global df
    if df is None:
        try:
            df = pd.read_csv("zz_test.csv")
        except:
            df = pd.DataFrame([], columns=['Image'])
    if not C_name in df.columns:
        df[C_name] = None
    row = df[df["Image"] == image_id]
    if(len(row) == 0):
        df2 = pd.DataFrame([[image_id, value]], columns=['Image', C_name])
        df = pd.concat([df, df2])
    else:
        df.loc[df["Image"] == image_id, C_name] = value

    if save:
        df = df.sort_values(by="Image")
        df.to_csv("zz_test.csv", index=False)

append_to_csv(C_name, image_id, value)
for image_id in range(10000):
    append_to_csv(C_name, image_id, random.random(), False)

df = df.sort_values(by="Image")
df.to_csv("zz_test.csv", index=False)

