import glob
import pandas as pd


filenames = glob.glob("feature labels/*.csv")
dfs = [pd.read_csv(filename) for filename in filenames]

df = pd.concat(dfs, axis=0)

# Only the first 250 observations were labeled.
df = df.sort_values("id")[:250]

# Category names were just for labeler reference.
df = df.rename(columns={"race_(b,w,a,l,m,o)": "race"})

# One-hot age, race.
df = pd.get_dummies(df, columns=["age_in_decades", "race"])

# Labelers forgot to include human=1 for observations where subject_ambiguous=1
df["human"] = df["human"].fillna(1)

# Set features where subject_ambiguous=1 to 0.
df = df.fillna(0)

df = df.astype(int)

print(df.to_string())

df.to_csv("feature_labels.csv", index=False)
