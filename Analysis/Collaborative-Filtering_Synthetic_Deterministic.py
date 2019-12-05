import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import surprise
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms import KNNWithMeans, NMF, SVD, Prediction
from tqdm import tqdm

# https://blog.cambridgespark.com/tutorial-practical-introduction-to-recommender-systems-dbe22848392b

np.random.seed(0)

df_swipes = pd.read_csv("../Data Simulation/synthetic_swipes.csv", index_col=None)

# Shuffle rows.
df_swipes = df_swipes.sample(frac=1)

names = [col for col in df_swipes.columns if col != "id"]
test_count = 50


def subswipe_data(df, n_train, n_test, n_users):
    np.random.seed(0)

    # Use only the first n users.
    users = [f"synthetic_{i}" for i in range(n_users)]

    # Create a sample with missing swipes. Number of swipes in training set is n_strain. Number in test set is n_test.
    train_ids = df["id"][n_test: n_test + n_train]
    test_ids = df["id"][:n_test]

    print('train: ', len(train_ids), 'test: ', len(test_ids))

    # Create train set.
    sparse = pd.DataFrame(columns=["uid", "iid", "swipe"])
    for user in users:
        for train_id in train_ids:
            sparse = sparse.append({
                "uid": user,
                "iid": train_id,
                "swipe": df[user].loc[train_id]
                },
                ignore_index=True)
    print(sparse.shape)
    reader = surprise.Reader(rating_scale=(0, 1))
    train = surprise.Dataset.load_from_df(sparse, reader)
    train, _ = train_test_split(train, test_size=1e-10, random_state=0)

    # Create test set.
    sparse = pd.DataFrame(columns=["uid", "iid", "swipe"])
    for user in users:
        for test_id in test_ids:
            sparse = sparse.append({
                "uid": user,
                "iid": test_id,
                "swipe": df[user].loc[test_id]
                },
                ignore_index=True)
    reader = surprise.Reader(rating_scale=(0, 1))
    test = surprise.Dataset.load_from_df(sparse, reader)
    _, test = train_test_split(test, train_size=1e-10, random_state=0)

    return train, test


def acc(df, alg, algname, n_train, n_users, cutoff=.5):
    np.random.seed(0)

    train, test = subswipe_data(df, n_train=n_train, n_test=test_count, n_users=n_users)

    alg.fit(train)
    predictions = alg.test(test)

    # Change predictions to binary choice of left or right. Prediction class derives from NamedTuple.
    predictions = [
        Prediction(prediction.uid, prediction.iid, prediction.r_ui, int(prediction.est < cutoff), prediction.details)
        for prediction in predictions
    ]

    # Mean absolute error.
    mae = accuracy.mae(predictions)

    df_predicted = pd.DataFrame(columns=["uid", "iid", "predicted", "actual"])
    for prediction in predictions:
        df_predicted = df_predicted.append(
            {
                "uid": prediction.uid,
                "iid": prediction.iid,
                "predicted": prediction.est,
                "actual": df_swipes[prediction.uid].loc[prediction.iid]
            },
            ignore_index=True
        )

    acc_dict = {"algname": algname, "n_train": n_train, "n_users": n_users, "acc": mae}
    print(acc_dict)
    return acc_dict


if __name__ == "__main__":
    # Save accuracy data for all swipe values and user values.
    df_acc = pd.DataFrame(columns=["algname", "n_train", "n_users", "acc"])

    max_swipes = 210
    for alg, algname in tqdm(zip([SVD(), NMF(), KNNWithMeans()], ["SVD", "NMF", "KNNWithMeans"])):
        for n_users in [5, 10, 25, 50, 100, 250, 500, 1000]:
            for n_train in range(10, max_swipes, 10):
                df_acc = df_acc.append(acc(df_swipes, alg, algname, n_train, n_users), ignore_index=True)
            df_acc.to_csv("acc_deterministic.csv", index=False)
