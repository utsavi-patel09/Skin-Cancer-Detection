import numpy as np
from keras import layers
import tensorflow as tf
from keras.layers import Input, Dense, Activation,BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle



file1 = 'D://SGP-SEM5//skin_cancer//train'
file2 = 'D://SGP-SEM5//skin_cancer//test'
image_generator = ImageDataGenerator(horizontal_flip=True,validation_split=0.33)
train_data_gen = image_generator.flow_from_directory(directory=file1, target_size=(227,227),batch_size=48, class_mode='categorical',subset='training')
val_data_gen = image_generator.flow_from_directory(directory=file2, target_size=(227,227),batch_size=48, class_mode='categorical',subset='training')

pretrained_model = tf.keras.applications.resnet50.ResNet50(
                    input_shape=(227, 227, 3),
                    include_top=False,
                    weights='imagenet',
                    pooling='avg')

pretrained_model.trainable = False

inputs = pretrained_model.input
x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dense(50, activation='relu')(x)
outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
print(model.summary())


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(train_data_gen,validation_data=val_data_gen,epochs=25)

with open("D://SGP-SEM5//skin_cancer//train//Resnet50_100.pkl", "wb") as f:
    pickle.dump(model, f)

