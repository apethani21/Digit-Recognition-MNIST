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


tf.random.set_seed(1)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


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


def log_confusion_matrix_higher_order(epoch,
                                      logs,
                                      log_dir,
                                      model,
                                      test_dataset,
                                      test_labels):
    file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')
    test_pred_raw = model.predict(test_dataset)
    test_pred = np.argmax(test_pred_raw, axis=1)
    cm = confusion_matrix(test_labels, test_pred)
    figure = plot_confusion_matrix(model, cm, test_dataset, labels=class_names)
    cm_image = plot_to_image(figure)
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)


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
    log_confusion_matrix = partial(log_confusion_matrix_higher_order, 
                                   log_dir=log_dir,
                                   model=model,
                                   test_dataset=test_dataset,
                                   test_labels=test_labels) #! GET test_labels
    cm_callback = (tf
                   .keras
                   .callbacks
                   .LambdaCallback(on_epoch_end=log_confusion_matrix))
    steps = int((num_train_examples/batch_size) + 1)
    history = model.fit(x=train_dataset,
                        epochs=epochs,
                        steps_per_epoch=steps,
                        callbacks=[model_checkpoint,
                                   tensorboard_callback,
                                   cm_callback])
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


if __name__ == "__main__":
    arg = sys.argv[1]
    model_name = arg.split('=')[-1]
    print(f"model name: {model_name}")
    os.makedirs(f'./.models/{model_name}', exist_ok=True)
    path_to_save = f'./.models/{model_name}/{model_name}.h5'
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)
    compute(model_name, batch_size=64, epochs=15)
