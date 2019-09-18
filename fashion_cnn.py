import pickle
import logging
import tensorflow as tf
import tensorflow_datasets as tfds


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


def get_model():
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', 
                           activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((3, 3), strides=2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(32, (3,3), padding='same', 
                           activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.MaxPooling2D((3, 3), strides=2),
    tf.keras.layers.Conv2D(32, (3,3), padding='same', 
                           activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10,  activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model


def train(batch_size=32, epochs=15):
    train_dataset, test_dataset, metadata = get_data()
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
                        .ModelCheckpoint('./config/fashion_cnn.h5', verbose=True)
    )
    global model
    model = get_model()
    steps = int((num_train_examples/batch_size) + 1)
    history = model.fit(x=train_dataset,
                        epochs=epochs,
                        steps_per_epoch=steps,
                        callbacks = [model_checkpoint])
    model.save('./config/fashion_cnn.h5')
    return history, steps, test_dataset


def evaluate_model(steps, test_dataset):
    acc = model.evaluate(test_dataset, steps=steps)
    print('Accuracy: ', acc)

    
def compute():
    history, steps, test_dataset = train(batch_size=64, epochs=10)
    with open('./config/history', 'wb') as f:
        pickle.dump(history.history, f)
    evaluate_model(steps, test_dataset)


if __name__ == "__main__":
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)
    compute()
