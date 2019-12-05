import pandas as pd

df = pd.read_csv("synthetic_swipes.csv")

users = [f"synthetic_{i}" for i in range(1000)]

avg_swipe = df[users].sum() / df[users].shape[0]

print('mean:', avg_swipe.mean(), "std:", avg_swipe.std())
