import numpy as np
import os
import pandas as pd
import skimage
from keras.layers import Convolution2D, GlobalAveragePooling2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Model, Sequential
from PIL import Image
from skimage.transform import resize
from tqdm import tqdm


def autoencoder(sequential):
    autoencoder = sequential
    # for layer in sequential.layers:
    #     print(layer)
    #     print(layer.name)
    #     print(type(layer))
    for layer in sequential.layers[::-1]:
        print(layer.name)


    for layer in sequential.layers[::-1]:
        print(layer.name)
        if "conv2d" in layer.name:
            autoencoder = UpSampling2D(autoencoder.output)
        if "input" in layer.name:
            pass
        else:
            autoencoder = layer(autoencoder.output)

    return autoencoder


# Set up image location and sizes.

image_path = "../Analysis/ladies1"
WIDTH = 1080
HEIGHT = 1080
DIMS = 3
K = list(range(4096))


# https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/

os.chdir("/Users/PycharmProjects/Tinder_Chad/VGG_FACE")


# Create the VGG FACE model.

def create_convolutional_mode():
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

    return convolutional_model


convolutional_model = create_convolutional_mode()

convolutional_model_4096_7 = Model(inputs=convolutional_model.inputs, outputs=GlobalAveragePooling2D()(convolutional_model.layers[-4].output))
convolutional_model_4096_7.summary()

convolutional_model_4096_1 = Model(inputs=convolutional_model.inputs, outputs=GlobalAveragePooling2D()(convolutional_model.layers[-3].output))
convolutional_model_4096_1.summary()

ae_7 = autoencoder(convolutional_model_4096_7)
ae_7.summary()
ae_1 = autoencoder(convolutional_model_4096_1)
ae_1.summary()

# Reconstruct images with autoencoder.

R = list(range(len([filename for filename in os.listdir(f"{image_path}/") if ".jpg" in filename])))

for id in tqdm(R):
    img = skimage.io.imread(f"{image_path}/{id}.jpg")
    img = resize(img, (WIDTH, HEIGHT))
    img = img.reshape((1, WIDTH, HEIGHT, DIMS))

    for ae in [ae_7, ae_1]:
        img = ae.predict(img)
        print(img.shape)
        img = Image.fromarray(img, 'RGB')
        img.save(f'img{id}_{ae}.png')
        img.show()
