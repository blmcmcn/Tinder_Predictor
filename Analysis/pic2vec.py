import numpy as np
import os
import pandas as pd
import skimage
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, GlobalAveragePooling2D
from keras.models import Model
from skimage.transform import resize


# Convert images to tensor.

os.chdir("/Users/PycharmProjects/Tinder_Chad/")
image_path = "ladies1"
WIDTH = 1080
HEIGHT = 1080
DIMS = 3

images = []
R = None

if "images.npy" not in os.listdir(f"Analysis/"):
    R = list(range(len([filename for filename in os.listdir(f"Analysis/{image_path}/") if ".jpg" in filename])))
    print(R)
    for i in R:
        try:
            img = skimage.io.imread(f"Analysis/{image_path}/{i}.jpg")
            img = resize(img, (WIDTH, HEIGHT))
            img = img.reshape([WIDTH, HEIGHT, DIMS])
            images.append(img)
        except ValueError:
            pass
    images = np.array(images)
    np.save("images.npy", images)
else:
    images = np.load(f"Analysis/images.npy")
    #images = images[:10, :, :, :]
    print(images.shape)
    R = list(range(images.shape[0]))
    print(R)

# Get feature vectors from convolutional network.

convolutional_model = InceptionV3(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(WIDTH, HEIGHT, DIMS),
)

output = GlobalAveragePooling2D()(convolutional_model.output)
convolutional_model = Model(inputs=convolutional_model.inputs, outputs=output)
convolutional_model.summary()

image_data = convolutional_model.predict(images)
print(image_data.shape)
image_data = pd.DataFrame(image_data, columns=range(image_data.shape[1]))
image_data["id"] = R

image_data.to_csv("image_data.csv", sep=',', index=False)

for col in image_data:
    print(col, image_data[col].mean(), image_data[col].std())

# Normalized flattened CNN features.


# Handle NaNs.
