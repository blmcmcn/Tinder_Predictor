import numpy as np
import os
import pandas as pd
import skimage
from keras.layers import Activation, Convolution2D, Dropout, Flatten, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D
from keras.models import Model, Sequential
from skimage.transform import resize
from tqdm import tqdm


# Set up image location and sizes.

image_path = "cropped_largest_resized"
WIDTH = 1074
HEIGHT = 1074
DIMS = 3
K = list(range(4096))


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
convolutional_model.add(Convolution2D(4096, (1, 1), activation='relu'))
convolutional_model.add(Convolution2D(2622, (1, 1)))
convolutional_model.add(GlobalAveragePooling2D())

convolutional_model.load_weights('vgg_face_weights.h5')


convolutional_model_4096_7 = Model(inputs=convolutional_model.inputs, outputs=GlobalAveragePooling2D()(convolutional_model.layers[-4].output))
convolutional_model_4096_7.summary()

convolutional_model_4096_1 = Model(inputs=convolutional_model.inputs, outputs=GlobalAveragePooling2D()(convolutional_model.layers[-3].output))
convolutional_model_4096_1.summary()


# Create .csv.

os.chdir("/Users/PycharmProjects/Tinder_Chad/")

R = list(range(len([filename for filename in os.listdir(f"Analysis/{image_path}/") if ".jpg" in filename])))
#R = [0, 1, 2, 3]

image_data_4096_7 = pd.DataFrame(columns=["id"] + K)
image_data_4096_1 = pd.DataFrame(columns=["id"] + K)

for id in tqdm(R):
    img = skimage.io.imread(f"Analysis/{image_path}/{id}.jpg")
    img = resize(img, (WIDTH, HEIGHT))
    img = img.reshape((1, WIDTH, HEIGHT, DIMS))

    # Get feature vectors from convolutional network.
    img_4096_7 = convolutional_model_4096_7.predict(img)
    img_4096_7 = pd.DataFrame(img_4096_7, columns=K)
    print(img_4096_7.shape)
    img_4096_7["id"] = id
    image_data_4096_7 = image_data_4096_7.append(img_4096_7)

    # Get feature vectors from convolutional network.
    img_4096_1 = convolutional_model_4096_1.predict(img)
    img_4096_1 = pd.DataFrame(img_4096_1, columns=K)
    print(img_4096_1.shape)
    img_4096_1["id"] = id
    image_data_4096_1 = image_data_4096_1.append(img_4096_1)

image_data_4096_7.to_csv("vgg_face_data_4096_7_cropped.csv", sep=',', index=False)
image_data_4096_1.to_csv("vgg_face_data_4096_1_cropped.csv", sep=',', index=False)

print(image_data_4096_7.shape)
print(image_data_4096_1.shape)
