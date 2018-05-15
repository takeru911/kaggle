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

def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
    model = applications.VGG16(include_top=False, weights="imagenet")

    generator = datagen.flow_from_directory(
        "input/images/train",
        target_size=(150, 150),
        batch_size=batch_size,
        subset='training'
        )
    bottleneck_features_train = model.predict_generator(
        generator,
        2000 // batch_size
        )
    np.save(
        "models/bottleneck_features_train.npy",
        bottleneck_features_train
            )
    generator = datagen.flow_from_directory(
        "input/images/train",
        target_size=(150, 150),
        batch_size=batch_size,
        subset='validation'
        )
    bottleneck_features_validation = model.predict_generator(
        generator, 800 // batch_size
        )
    np.save(
        "models/bottleneck_features_validation.npy",
        bottleneck_features_validation
        )

save_bottleneck_features()
train_data = np.load("bottleneck_features_train.npy")
train_labels = np.array([])


