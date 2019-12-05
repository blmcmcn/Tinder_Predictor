import numpy as np
import pandas as pd
import surprise
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms import KNNWithMeans, NMF, SVD, Prediction

# https://blog.cambridgespark.com/tutorial-practical-introduction-to-recommender-systems-dbe22848392b

np.random.seed(0)

swipes = pd.read_csv("data/combined.csv", index_col=None)
names = [col for col in swipes.columns if col != "id"]
"""
people used vs accuracy:
3: .502
4: .604
5: .616
6: .713
"""

# Create a sample with missing swipes. Number of swipes in dataset is n_subswipe.
n_subswipe = 600
subswipe_ids = {name: np.random.choice(swipes["id"], n_subswipe, replace=False) for name in names}
complement_ids = {name: set(swipes["id"]) - set(subswipe_ids[name]) for name in names}
sparse = pd.DataFrame(columns=["uid", "iid", "swipe"])
for name in names:
    for subswipe_id in subswipe_ids[name]:
        sparse = sparse.append({"uid": name, "iid": subswipe_id, "swipe": swipes[name].loc[subswipe_id]}, ignore_index=True)

# print(sparse)

reader = surprise.Reader(rating_scale=(0, 1))
data = surprise.Dataset.load_from_df(sparse, reader)

'''
for alg in [SVD(), NMF(), KNNWithMeans()]:
    output = alg.fit(data.build_full_trainset())

    preds={}
    for name in names:
        preds[name] = sorted([(i, alg.predict(uid=name, iid=str(i)).est) for i in complement_ids[name]], key=lambda x: x[1], reverse=True)

    print(preds)
'''

cutoff = .5
trainset, testset = train_test_split(data, test_size=0.25)
for alg in [SVD(), NMF(), KNNWithMeans()]:
    alg.fit(trainset)
    predictions = alg.test(testset)

    # Change predictions to binary choice of left or right. Prediction class derives from NamedTuple.
    predictions = [
        Prediction(prediction.uid, prediction.iid, prediction.r_ui, int(prediction.est < cutoff), prediction.details)
        for prediction in predictions
    ]
    # print(predictions)
    accuracy.mae(predictions)

    df_predicted = pd.DataFrame(columns=["uid", "iid", "predicted", "actual"])
    for prediction in predictions:
        df_predicted = df_predicted.append(
            {
                "uid": prediction.uid,
                "iid": prediction.iid,
                "predicted": prediction.est,
                "actual": swipes[prediction.uid].loc[prediction.iid]
            },
            ignore_index=True
        )

    # print(df_predicted.to_string())
    print("my_acc: ", abs(df_predicted["predicted"] - df_predicted["actual"]).sum() / df_predicted.shape[0])
