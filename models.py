import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def model_one():
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), padding='same',
                      activation=tf.nn.relu, input_shape=(28, 28, 1)),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Dropout(0.2),
        layers.Conv2D(32, (3,3), padding='same',
                      activation=tf.nn.relu),
        layers.Dropout(0.2),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Conv2D(32, (3,3), padding='same',
                      activation=tf.nn.relu),
        layers.MaxPooling2D((2, 2), strides=2),
        layers.Flatten(),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def model_two():
    model = keras.Sequential([
        layers.Conv2D(64, (3,3), padding='same',
                      activation=tf.nn.relu, input_shape=(28, 28, 1)),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Dropout(0.2),
        layers.Conv2D(64, (3,3), padding='same',
                      activation=tf.nn.relu),
        layers.Dropout(0.2),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Conv2D(64, (3,3), padding='same',
                      activation=tf.nn.relu),
        layers.MaxPooling2D((2, 2), strides=2),
        layers.Flatten(),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model