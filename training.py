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




# Set paths for training and test datasets
train_dir = 'D://SGP-SEM5//skin_cancer//train'
test_dir = 'D://SGP-SEM5//skin_cancer//test'

# Data Augmentation using ImageDataGenerator
image_generator = ImageDataGenerator(horizontal_flip=True, validation_split=0.33)

# Load the training and validation data
train_data_gen = image_generator.flow_from_directory(
    directory=train_dir, 
    target_size=(227, 227), 
    batch_size=48, 
    class_mode='categorical', 
    subset='training'
)

val_data_gen = image_generator.flow_from_directory(
    directory=test_dir, 
    target_size=(227, 227), 
    batch_size=48, 
    class_mode='categorical', 
    subset='training'
)

# Load the VGG16 model, excluding the top layers and using pre-trained weights
pretrained_model = tf.keras.applications.VGG16(
    input_shape=(227, 227, 3),
    include_top=False,  # Do not include the fully-connected layers on top
    weights='imagenet',
    pooling='avg'
)

# Freeze the pretrained model layers to prevent training
pretrained_model.trainable = False

# Add custom layers on top of VGG16
inputs = pretrained_model.input
x = Dense(128, activation='relu')(pretrained_model.output)
x = Dense(50, activation='relu')(x)
outputs = Dense(2, activation='softmax')(x)  # 2 output neurons for binary classification (cancer vs. no cancer)

# Build the final model
model = Model(inputs, outputs)
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data_gen, 
    validation_data=val_data_gen, 
    epochs=10
)



with open("D://SGP-SEM5//skin_cancer//train//vgg16.pkl", "wb") as f:
    pickle.dump(model, f)

