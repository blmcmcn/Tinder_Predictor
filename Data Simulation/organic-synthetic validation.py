import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df_organic = pd.read_csv("../Analysis/data/combined.csv")
df_feature = pd.read_csv("feature_labels.csv")

N = 250

users = [col for col in df_organic.columns if col != "id"]
features = [col for col in df_feature.columns if col != "id"]

print(users)
print(features)

for user in users:
    X = df_feature[features].iloc[:N]
    y = df_organic[user].iloc[:N]
    model = LogisticRegression()
    model.fit(X, y)
    p_hat = model.predict_proba(X)[:, 1]
    thresholds = np.linspace(0, 1, 1000)
    accs = [accuracy_score(y_true=y, y_pred=(p_hat > threshold).astype(int)) for threshold in thresholds]
    acc_max = max(accs)
    print(user, acc_max)

    coef = model.coef_[0]
    w = list(zip(coef, features))
    w.sort(key=lambda x: abs(x[0]), reverse=True)
    print(user, w)