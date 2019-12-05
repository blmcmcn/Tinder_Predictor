import numpy as np
import os
import pandas as pd
import skimage
from keras.layers import Activation, Convolution2D, Dropout, Flatten, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D
from keras.models import Model, Sequential
from skimage.transform import resize
from tqdm import tqdm


# Set up image location and sizes.

image_path = "ladies1"
WIDTH = 1080
HEIGHT = 1080
DIMS = 3
K = list(range(2622))


# https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/

# Create the VGG FACE model.

convolutional_model = Sequential()
convolutional_model.add(ZeroPadding2D((1, 1), input_shape=(WIDTH, HEIGHT, DIMS)))
convolutional_model.add(Convolution2D(64, (3, 3), activation='relu'))
convolutional_model.add(ZeroPadding2D((1, 1)))
convolutional_model.add(Convolution2D(64, (3, 3), activation='relu'))
convolutional_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

convolutional_model.add(ZeroPadding2D((1, 1)))
convolutional_model.add(Convolution2D(128, (3, 3), activation='relu'))
convolutional_model.add(ZeroPadding2D((1, 1)))
convolutional_model.add(Convolution2D(128, (3, 3), activation='relu'))
convolutional_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

convolutional_model.add(ZeroPadding2D((1, 1)))
convolutional_model.add(Convolution2D(256, (3, 3), activation='relu'))
convolutional_model.add(ZeroPadding2D((1, 1)))
convolutional_model.add(Convolution2D(256, (3, 3), activation='relu'))
convolutional_model.add(ZeroPadding2D((1, 1)))
convolutional_model.add(Convolution2D(256, (3, 3), activation='relu'))
convolutional_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

convolutional_model.add(ZeroPadding2D((1, 1)))
convolutional_model.add(Convolution2D(512, (3, 3), activation='relu'))
convolutional_model.add(ZeroPadding2D((1, 1)))
convolutional_model.add(Convolution2D(512, (3, 3), activation='relu'))
convolutional_model.add(ZeroPadding2D((1, 1)))
convolutional_model.add(Convolution2D(512, (3, 3), activation='relu'))
convolutional_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

convolutional_model.add(ZeroPadding2D((1, 1)))
convolutional_model.add(Convolution2D(512, (3, 3), activation='relu'))
convolutional_model.add(ZeroPadding2D((1, 1)))
convolutional_model.add(Convolution2D(512, (3, 3), activation='relu'))
convolutional_model.add(ZeroPadding2D((1, 1)))
convolutional_model.add(Convolution2D(512, (3, 3), activation='relu'))
convolutional_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

convolutional_model.add(Convolution2D(4096, (7, 7), activation='relu'))
convolutional_model.add(Dropout(0.5))
convolutional_model.add(Convolution2D(4096, (1, 1), activation='relu'))
convolutional_model.add(Dropout(0.5))
convolutional_model.add(Convolution2D(2622, (1, 1)))
convolutional_model.add(GlobalAveragePooling2D())

convolutional_model.load_weights('vgg_face_weights.h5')


# convolutional_model = Model(inputs=convolutional_model.inputs, outputs=convolutional_model.layers[-2].output)
# convolutional_model.summary()


# Create .csv.

os.chdir("/Users/PycharmProjects/Tinder_Chad/")

R = list(range(len([filename for filename in os.listdir(f"Analysis/{image_path}/") if ".jpg" in filename])))
#R = [0, 1, 2, 3]

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

image_data.to_csv("vgg_face_data.csv", sep=',', index=False)

print(image_data.shape)
