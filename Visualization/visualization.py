import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("acc.csv")
for col in df.columns:
    if col != "algname" and col != "acc":
        df[col] = df[col].astype(int)

df_SVD = df[df["algname"] == "SVD"]


plt.figure()
plt.scatter(df_SVD["n_train"], np.log(df_SVD["n_users"]), s=1000 * (df_SVD["acc"] - .7))
plt.show()

plt.figure()
# for n_users in sorted(list(set(df_SVD["n_users"]))):
for n_users in [50, 100, 250, 500, 1000]:
    df_n_users = df_SVD[df["n_users"] == n_users]
    plt.plot(df_n_users["n_train"], df_n_users["acc"], label=f"number of users = {n_users}")
plt.legend(loc='best')
plt.title("Collaborative Filtering with SVD")
plt.xlabel("number of swipes")
plt.ylabel("accuracy")
plt.show()
