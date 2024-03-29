import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from tqdm import tqdm


np.random.seed(0)

df_raw = pd.read_csv("image_data.csv")
df_face = pd.read_csv("vgg_face_data_4096_7.csv")
df_swipe = pd.read_csv("user1.csv")

# Reaname features for pic2vec to appropriate embedding scheme.
df_raw.columns = [f"raw{col}" if isinstance(col, int) else col for col in df_raw.columns]
df_face.columns = [f"face{col}" if isinstance(col, int) else col for col in df_face.columns]

# Merge raw, face, and swipe data.
df_swipe = pd.merge(df_raw, df_swipe, how="inner", on="id")
df_swipe = pd.merge(df_face, df_swipe, how="inner", on="id")

# Make swipe a binary variable.
df_swipe["swipe"] = df_swipe["swipe"] == "r"

print("percent right: ", df_swipe["swipe"].sum() / df_swipe.shape[0])

datasets = "train", "validate", "test"
df = {}
X = {}
Y = {}

df["train"], df["validate"], df["test"] = np.split(df_swipe.sample(frac=1), [int(.6 * len(df_swipe)), int(.8 * len(df_swipe))])

for dataset in datasets:
    X[dataset] = df[dataset][[col for col in df_swipe.columns if col not in ["id", "swipe"]]].as_matrix()
    Y[dataset] = df[dataset]["swipe"].as_matrix()
    print(X[dataset].shape, Y[dataset].shape)


input_shape = (X["train"].shape[1],)


# Show how model accuracy increases as a function of N.

N_train = X["train"].shape[0]
accuracy_list = [0] * N_train

for i in tqdm(range(2, N_train)):
    X_train = X["train"][:i, :]
    Y_train = Y["train"][:i]
    model = xgb.XGBClassifier(random_state=0)
    model.fit(X_train, Y_train)
    Y_hat = model.predict(X["validate"])
    accuracy_list[i] = accuracy_score(y_true=Y["validate"], y_pred=Y_hat)

plt.figure()
plt.title("XGB accuracy versus N (raw + face_7 features)")
plt.scatter(range(len(accuracy_list)), accuracy_list)
plt.show()
