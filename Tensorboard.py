# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from cv2 import cv2
import random

import pydot
import graphviz

import time

NameX = "X_MIXED.pickle"
NameY = "y_MIXED.pickle"

X = pickle.load(open(NameX, "rb"))
y = pickle.load(open(NameY, "rb"))

X = np.array(X) / 255.0
y = np.array(y)

image_size = (500, 500)
batch_size = 32
# onii-chan

NAME = "Classic-100-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='Classic/logs/{}'.format(NAME))

model = Sequential()
model.add(  Conv2D(64, (3,3), input_shape = (500, 500, 1))  )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(  Conv2D(64, (3,3))  )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(  Conv2D(64, (3,3))  )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
 
model.add(Flatten())

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
optimizer="adam", metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorboard])

model.save("keras-Classic")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(r"C:\Users\chern\androidRepository\NNModels\Classic", "xb").write(tflite_model)