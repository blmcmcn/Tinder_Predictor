import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Activation, Dense, LeakyReLU
from sklearn.metrics import accuracy_score


np.random.seed(0)

df_image = pd.read_csv("image_data.csv")
df_swipe = pd.read_csv("user1.csv")

# Merge image and swipe data.
df_swipe = pd.merge(df_image, df_swipe, how="inner", on="id")

# Make swipe a binary variable.
df_swipe["swipe"] = df_swipe["swipe"] == "r"



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


def get_model(layers):
    model = Sequential()
    model.add(Dense(input_shape=input_shape, units=layers[0]))
    model.add(Activation("tanh"))
    for layer in layers[1:]:
        model.add(Dense(units=layer))
        model.add(LeakyReLU(alpha=.1))
    model.add(Dense(units=1))
    model.add(Activation("softmax"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


# Show how model accuracy increases as a function of N.

N_train = X["train"].shape[0]
accuracy_list = [0] * N_train

for i in range(1, N_train):
    X_train = X["train"][:i, :]
    Y_train = Y["train"][:i]
    model = get_model([100, 100, 50])
    model.fit(X_train, Y_train, batch_size=32, epochs=1)
    Y_hat = model.predict(X["validate"])
    accuracy_list[i] = accuracy_score(y_true=Y["validate"], y_pred=Y_hat)

plt.figure()
plt.scatter(range(len(accuracy_list)), accuracy_list)
plt.show()
