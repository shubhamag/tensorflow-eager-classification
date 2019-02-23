from __future__ import absolute_import, division, print_function

import argparse

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib import eager as tfe
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
        self.dense2 = layers.Dense(num_classes)

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


def preprocess_data(images, labels):
    images = images.astype('float32') / 255.0
    images = images.reshape(images.shape[0], 28, 28, 1)
    labels = labels.astype('int32')
    return images, labels


def test(model, dataset):
    test_loss = tfe.metrics.Mean()
    test_accuracy = tfe.metrics.Accuracy()
    for batch, (images, labels) in enumerate(dataset):
        logits = model(images)
        loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, labels)
        test_loss(loss_value)
    return test_loss.result(), test_accuracy.result()


def predict(model, images, class_names):
    predictions = model(images)
    for i, logits in enumerate(predictions):
        class_idx = tf.argmax(logits).numpy()
        p = tf.nn.softmax(logits)[class_idx]
        name = class_names[class_idx]
        print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100 * p))


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
    parser.add_argument('--log-dir', type=str, default='tb',
                        help='path for logging directory')
    parser.add_argument('--ckpt-dir', type=str, default='ckpt',
                        help='path for saving model')

    args = parser.parse_args()
    start_time = time.time()
    util.set_random_seed(args.seed)
    sess = util.set_session()

    fashion_mnist = keras.datasets.fashion_mnist
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images, train_labels = preprocess_data(train_images,
                                                 train_labels)
    test_images, test_labels = preprocess_data(test_images,
                                               test_labels)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.shuffle(10000).batch(args.batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_dataset = test_dataset.batch(args.batch_size)

    model = SimpleCNN(num_classes=len(class_names))

    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
    train_log = {'iter': [], 'loss': [], 'accuracy': []}
    test_log = {'iter': [], 'loss': [], 'accuracy': []}
    for ep in range(args.epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()
        for batch, (images, labels) in enumerate(train_dataset):
            loss_value, grads = util.cal_grad(model,
                                              loss_func=tf.losses.sparse_softmax_cross_entropy,
                                              inputs=images,
                                              targets=labels)
            optimizer.apply_gradients(zip(grads,
                                          model.trainable_variables),
                                      global_step)
            epoch_loss_avg(loss_value)
            epoch_accuracy(tf.argmax(model(images),
                                     axis=1,
                                     output_type=tf.int32),
                           labels)
            if global_step.numpy() % args.log_interval == 0:
                print('Epoch: {0:d}/{1:d} Iteration:{2:d}  Training Loss:{3:.4f}  '
                      'Training Accuracy:{4:.4f}'.format(ep,
                                                         args.epochs,
                                                         global_step.numpy(),
                                                         epoch_loss_avg.result(),
                                                         epoch_accuracy.result()))
                train_log['iter'].append(global_step.numpy())
                train_log['loss'].append(epoch_loss_avg.result())
                train_log['accuracy'].append(epoch_accuracy.result())
            if global_step.numpy() % args.eval_interval == 0:
                test_loss, test_acc = test(model, test_dataset)
                test_log['iter'].append(global_step.numpy())
                test_log['loss'].append(test_loss)
                test_log['accuracy'].append(test_acc)

    model.summary()
    end_time = time.time()
    print('Elapsed time: {0:.3f}s'.format(end_time - start_time))
    predict(model, test_images[:5], class_names)
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
    tf.enable_eager_execution()
    main()
