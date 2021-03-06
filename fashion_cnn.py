import os
import io
import sys
import pickle
import logging
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

logger = tf.get_logger()
logger.setLevel(logging.INFO)
tf.random.set_seed(1)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def get_data():
    dataset, metadata = tfds.load('fashion_mnist',
                                  as_supervised=True,
                                  with_info=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    return train_dataset, test_dataset, metadata


def normalise(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


def train(model_name, batch_size=32, epochs=15):
    model = getattr(__import__('models'), model_name)()
    print(model.summary())
    train_dataset, test_dataset, metadata = get_data()
    print(metadata)
    num_train_examples = metadata.splits['train'].num_examples
    train_dataset = (train_dataset
                     .map(normalise)
                     .repeat()
                     .shuffle(num_train_examples)
                     .batch(batch_size))
    test_dataset = (test_dataset
                    .map(normalise)
                    .batch(batch_size))
    print("Data prepared")
    path_to_save = f'./.models/{model_name}/{model_name}.h5'
    model_checkpoint = (tf
                        .keras
                        .callbacks
                        .ModelCheckpoint(path_to_save,
                                         verbose=True))
    log_dir = f".models/{model_name}/tb_logs"
    tensorboard_callback = (tf
                            .keras
                            .callbacks
                            .TensorBoard(log_dir=log_dir + '/fit',
                                         histogram_freq=1))
    steps = int((num_train_examples/batch_size) + 1)
    history = model.fit(x=train_dataset,
                        epochs=epochs,
                        steps_per_epoch=steps,
                        callbacks=[model_checkpoint,
                                   tensorboard_callback])
    model.save(path_to_save)
    return model, history, steps, test_dataset


def evaluate_model(model, steps, test_dataset):
    acc = model.evaluate(test_dataset, steps=steps)
    print('Accuracy: ', acc)


def compute(model_name, batch_size=64, epochs=10):
    (model,
     history,
     steps,
     test_dataset) = train(model_name, batch_size, epochs)
    with open(f'./.models/{model_name}/history', 'wb') as f:
        pickle.dump(history.history, f)
    evaluate_model(model, steps, test_dataset)


def main():
    arg = sys.argv[1]
    model_name = arg.split('=')[-1]
    print(f"model name: {model_name}")
    os.makedirs(f'./.models/{model_name}', exist_ok=True)
    compute(model_name, batch_size=64, epochs=15)


if __name__ == "__main__":
    main()
