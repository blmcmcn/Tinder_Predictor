import numpy as np
import os
import pandas as pd
import skimage
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, GlobalAveragePooling2D
from keras.models import Model
from skimage.transform import resize
from tqdm import tqdm


# Set up image location and sizes.

os.chdir("/Users/PycharmProjects/Tinder_Chad/")
image_path = "ladies1"
WIDTH = 1080
HEIGHT = 1080
DIMS = 3
K = list(range(2048))

R = list(range(len([filename for filename in os.listdir(f"Analysis/{image_path}/") if ".jpg" in filename])))


# Create convolutional network.

convolutional_model = InceptionV3(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(WIDTH, HEIGHT, DIMS),
)

output = GlobalAveragePooling2D()(convolutional_model.output)
convolutional_model = Model(inputs=convolutional_model.inputs, outputs=output)
convolutional_model.summary()


# Create .csv.

image_data = pd.DataFrame(columns=["id"]+K)

if "images.npy" not in os.listdir(f"Analysis/"):
    for id in tqdm(R):
        img = skimage.io.imread(f"Analysis/{image_path}/{id}.jpg")
        img = resize(img, (WIDTH, HEIGHT))
        img = img.reshape((1, WIDTH, HEIGHT, DIMS))

        # Get feature vectors from convolutional network.
        img = convolutional_model.predict(img)
        img = pd.DataFrame(img, columns=K)
        img["id"] = id
        image_data = image_data.append(img)

image_data.to_csv("image_data.csv", sep=',', index=False)

print(image_data.shape)


for col in image_data:
    print(col, image_data[col].mean(), image_data[col].std())
