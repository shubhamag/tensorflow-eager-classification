from __future__ import absolute_import, division, print_function

import argparse

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import util


class SimpleCNN(keras.Model):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__(name='SimpleCNN')
        self.num_classes = num_classes
        self.conv1 = layers.Conv2D(input_shape=(28, 28, 1),
                                   filters=32,
                                   kernel_size=[5, 5],
                                   padding="same",
                                   activation='relu')
        self.pool1 = layers.MaxPool2D(pool_size=(2, 2))
        self.conv2 = layers.Conv2D(filters=64,
                                   kernel_size=[5, 5],
                                   padding="same",
                                   activation='relu')
        self.pool2 = layers.MaxPool2D(pool_size=(2, 2))

        self.flat = layers.Flatten()

        self.dense1 = layers.Dense(1024, activation='relu')
        self.dropout = layers.Dropout(rate=0.4)
        self.dense2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        flat_x = self.flat(x)
        out = self.dense1(flat_x)
        out = self.dropout(out, training=training)
        out = self.dense2(out)
        return out

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape = [shape[0], self.num_classes]
        return tf.TensorShape(shape)


def preprocess_data(images, labels, num_classes=10):
    images = images.astype('float32') / 255.0
    images = images.reshape(images.shape[0], 28, 28, 1)
    labels = tf.keras.utils.to_categorical(labels, num_classes)
    return images, labels


def main():
    parser = argparse.ArgumentParser(description='TensorFlow Fashion MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=4,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=5,
                        help='how many batches to wait before'
                             ' logging training status')
    parser.add_argument('--eval-interval', type=int, default=100,
                        help='how many batches to wait before'
                             ' evaluate the model')

    args = parser.parse_args()
    start_time = time.time()
    util.set_random_seed(args.seed)
    sess = util.set_session()

    fashion_mnist = keras.datasets.fashion_mnist
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images, train_labels = preprocess_data(train_images,
                                                 train_labels,
                                                 num_classes=len(class_names))
    test_images, test_labels = preprocess_data(test_images,
                                               test_labels,
                                               num_classes=len(class_names))

    features_placeholder = tf.placeholder(train_images.dtype, train_images.shape)
    labels_placeholder = tf.placeholder(train_labels.dtype, train_labels.shape)
    train_dataset = tf.data.Dataset.from_tensor_slices((features_placeholder,
                                                        labels_placeholder))
    train_dataset = train_dataset.shuffle(10000).batch(args.batch_size)
    iterator = train_dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    model = SimpleCNN(num_classes=len(class_names))

    model.compile(optimizer=keras.optimizers.Adam(lr=args.lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    iter = 0
    train_log = {'iter': [], 'loss': [], 'accuracy': []}
    test_log = {'iter': [], 'loss': [], 'accuracy': []}
    for ep in range(args.epochs):
        sess.run(iterator.initializer,
                 feed_dict={features_placeholder: train_images,
                            labels_placeholder: train_labels})
        try:
            while True:
                iter += 1
                images, labels = sess.run(next_element)
                train_loss, train_acc = model.train_on_batch(images, labels)

                if iter % args.log_interval == 0:
                    print('Epoch: {0:d}/{1:d} Iteration:{2:d}  Training Loss:{3:.4f}  '
                          'Training Accuracy:{4:.4f}'.format(ep,
                                                             args.epochs,
                                                             iter,
                                                             train_loss,
                                                             train_acc))
                    train_log['iter'].append(iter)
                    train_log['loss'].append(train_loss)
                    train_log['accuracy'].append(train_acc)
                if iter % args.eval_interval == 0:
                    test_loss, test_acc = model.evaluate(test_images, test_labels)
                    test_log['iter'].append(iter)
                    test_log['loss'].append(test_loss)
                    test_log['accuracy'].append(test_acc)

        except tf.errors.OutOfRangeError:
            pass
    model.summary()
    end_time = time.time()
    print('Elapsed time: {0:.3f}s'.format(end_time - start_time))
    fig = plt.figure()
    plt.plot(train_log['iter'], train_log['loss'], 'r', label='Training')
    plt.plot(test_log['iter'], test_log['loss'], 'b', label='Testing')
    plt.title('Loss')
    plt.legend()
    fig = plt.figure()
    plt.plot(train_log['iter'], train_log['accuracy'], 'r', label='Training')
    plt.plot(test_log['iter'], test_log['accuracy'], 'b', label='Testing')
    plt.title('Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
