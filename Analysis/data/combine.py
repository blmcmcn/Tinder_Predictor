import pandas as pd

names = ["user1", "user2", "user3", "user4", "user5", "user6"]
dfs = [pd.read_csv(f"{name}.csv") for name in names]

for name, df in zip(names, dfs):
    df.columns = ["id", name]

combined = dfs[0]
for name, df in list(zip(names, dfs))[1:]:
    combined[name] = df[name]

combined = combined.replace(to_replace='r', value=1)
combined = combined.replace(to_replace='l', value=0)

print(combined)
print(combined.shape)

combined.to_csv("combined.csv", index=False)
