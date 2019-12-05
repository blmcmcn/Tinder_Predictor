import numpy as np
import pandas as pd

df = pd.read_csv("data/combined.csv")

means = []

for user in df.columns:
    print(user, df[user].mean())
    means.append(df[user].mean())

means = means[1:]
print("mean: ", np.mean(means))
print("std: ", np.std(means))
