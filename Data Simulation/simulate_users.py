import numpy as np
import pandas as pd


def random_user_swipes(X, mean=0, std=1):

    threshold = std * np.random.randn() + mean

    W = np.random.randn(X.shape[1], 1)

    W[0] += 1  # human
    W[1] = -1  # subject_ambiguous

    swipes = 1 / (1 + np.exp(X @ W)) < threshold

    #print(pd.Series(swipes.reshape(-1)))

    return pd.Series(swipes.reshape(-1))


if __name__ == "__main__":
    np.random.seed()

    df_features = pd.read_csv("feature_labels.csv")

    print(df_features.to_string())

    n_synthetic = 1000
    synthetic_swipes = pd.DataFrame({f"synthetic_{i}": random_user_swipes(df_features.drop("id", axis=1).values, mean=.5, std=.1)
                                     for i in range(n_synthetic)}, index=df_features["id"])

    synthetic_swipes["id"] = df_features["id"]

    synthetic_swipes = synthetic_swipes.astype(int)

    print(synthetic_swipes.to_string())

    synthetic_swipes.to_csv("synthetic_swipes.csv", index=False)