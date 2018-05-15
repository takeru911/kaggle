import numpy as np
import tensorflow as tf

from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Input, Dense
from keras.models import Model
from keras import applications

batch_size = 16


inputs = Input(shape=(150, 150, 3))
x = Conv2D(32, (3, 3))(inputs)
x = Activation("relu")(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(32, (3, 3))(x)
x = Activation("relu")(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (3, 3))(x)
x = Activation("relu")(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.5)(x)
prediction = Dense(55, activation="sigmoid")(x)

model = Model(inputs=inputs, outputs=prediction)
model.compile(
    loss="binary_crossentropy",
    optimizer="rmsprop",
    metrics=["accuracy"]
    )
generator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1
    )

train_generator = generator.flow_from_directory(
    "./input/images/train",
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode="categorical",
    subset='training'
    )

validation_generator = generator.flow_from_directory(
    "./input/images/train",
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode="categorical",
    subset='validation'
    )

model.fit_generator(
    train_generator,
    steps_per_epoch=50,
    epochs=2,
    validation_data=validation_generator,
    validation_steps=10
    )
model.save_weights("first_try.h5")
