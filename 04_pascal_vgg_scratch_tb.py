from __future__ import absolute_import, division, print_function

import argparse
import os
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.contrib import eager as tfe

import util

CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

class VGG(keras.Model):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__(name='VGG')
        self.num_classes = num_classes
        self.conv1_1 = layers.Conv2D(filters=64,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.conv1_2 = layers.Conv2D(filters=64,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation='relu')
        self.pool1 = layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2)
        
        self.conv2_1 = layers.Conv2D(filters=128,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation='relu')
        self.conv2_2 = layers.Conv2D(filters=128,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation='relu')
        self.pool2 = layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2)
        
        self.conv3_1 = layers.Conv2D(filters=256,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation='relu')
        self.conv3_2 = layers.Conv2D(filters=256,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation='relu')
        self.conv3_3 = layers.Conv2D(filters=256,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation='relu')
        self.pool3 = layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2)
        
        self.conv4_1 = layers.Conv2D(filters=512,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation='relu')
        self.conv4_2 = layers.Conv2D(filters=512,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation='relu')
        self.conv4_3 = layers.Conv2D(filters=512,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation='relu')
        self.pool4 = layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2)

        self.conv5_1 = layers.Conv2D(filters=512,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation='relu')
        self.conv5_2 = layers.Conv2D(filters=512,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation='relu')
        self.conv5_3 = layers.Conv2D(filters=512,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation='relu')
        self.pool5 = layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2)

        self.flat = layers.Flatten()

        self.dense1 = layers.Dense(4096, activation='relu')
        self.dropout1 = layers.Dropout(rate=0.5)
        self.dense2 = layers.Dense(4096, activation='relu')
        self.dropout2 = layers.Dropout(rate=0.5)
        self.dense3 = layers.Dense(num_classes)

    def call(self, inputs, training=True):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)

        flat_x = self.flat(x)
        out = self.dense1(flat_x)
        out = self.dropout1(out, training=training)
        out = self.dense2(out)
        out = self.dropout2(out, training=training)
        out = self.dense3(out)
        return out

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape = [shape[0], self.num_classes]
        return tf.TensorShape(shape)

def augment_train_data(x, y, z):
    x = tf.image.random_crop(x, size=(224, 224, 3))
    x = tf.image.random_flip_left_right(x)
    return x, y, z

def center_crop_test_data(x, y, z):
    x = tf.image.central_crop(x, central_fraction=0.875)
    return x, y, z

def test(model, dataset):
    test_loss = tfe.metrics.Mean()
    test_accuracy = tfe.metrics.Accuracy()
    for batch, (images, labels, weights) in enumerate(dataset):
        logits = model(images, training=False)
        loss_value = tf.losses.sigmoid_cross_entropy(labels, logits, weights)
        prediction = tf.round(tf.nn.sigmoid(logits))
        prediction = tf.cast(prediction, tf.int32)
        # prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, labels)
        test_loss(loss_value)
    return test_loss.result(), test_accuracy.result()



def main():
    parser = argparse.ArgumentParser(description='TensorFlow Pascal Example')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=60,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='how many batches to wait before'
                             ' logging training status')
    parser.add_argument('--eval-interval', type=int, default=250,
                        help='how many batches to wait before'
                             ' evaluate the model')
    parser.add_argument('--log-dir', type=str, default='tb',
                        help='path for logging directory')
    parser.add_argument('--data-dir', type=str, default='./data/VOCdevkit/VOC2007',
                        help='Path to PASCAL data storage')
    args = parser.parse_args()
    util.set_random_seed(args.seed)
    sess = util.set_session()

    train_images, train_labels, train_weights = util.load_pascal(args.data_dir,
                                                                 class_names=CLASS_NAMES,
                                                                 split='trainval')
    test_images, test_labels, test_weights = util.load_pascal(args.data_dir,
                                                              class_names=CLASS_NAMES,
                                                              split='test')


    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels, train_weights))
    train_dataset = train_dataset.map(augment_train_data)
    train_dataset = train_dataset.shuffle(10000).batch(args.batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels, test_weights))
    test_dataset = test_dataset.map(center_crop_test_data)
    test_dataset = test_dataset.batch(args.batch_size)

    model = VGG(num_classes=len(CLASS_NAMES))

    logdir = os.path.join(args.log_dir,
                          datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)
    writer = tf.contrib.summary.create_file_writer(logdir)
    writer.set_as_default()

    tf.contrib.summary.initialize()

    global_step = tf.train.get_or_create_global_step()

    decayed_lr = tf.train.exponential_decay(args.lr, global_step, 5000, 0.5, staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate=decayed_lr(), momentum=0.9)

    root = tf.train.Checkpoint(optimizer=optimizer,
                               model=model)

    train_log = {'iter': [], 'loss': [], 'accuracy': []}
    test_log = {'iter': [], 'loss': [], 'accuracy': []}

    ckpt_dir = 'pascal_vgg_2'
    ckpt_prefix = os.path.join(ckpt_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    
    for ep in range(args.epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        for batch, (images, labels, weights) in enumerate(train_dataset):
            loss_value, grads = util.cal_grad(model,
                                              loss_func=tf.losses.sigmoid_cross_entropy,
                                              inputs=images,
                                              targets=labels,
                                              weights=weights)

            grads_and_vars = zip(grads, model.trainable_variables)
            optimizer.apply_gradients(grads_and_vars, global_step)
            
            epoch_loss_avg(loss_value)
            if global_step.numpy() % args.log_interval == 0:
                print('Epoch: {0:d}/{1:d} Iteration:{2:d}  Training Loss:{3:.4f}'.format(ep,
                                                                                  args.epochs,
                                                                                  global_step.numpy(),
                                                                                  epoch_loss_avg.result()))
                train_log['iter'].append(global_step.numpy())
                train_log['loss'].append(epoch_loss_avg.result())
                
                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('Training Loss', loss_value)
                    tf.contrib.summary.image('RGB', images)
                    tf.contrib.summary.scalar('LR', decayed_lr())
                    
                    # print(len(grads_and_vars))
                    for i, variable in enumerate(model.trainable_variables):
                        tf.contrib.summary.histogram("grad_" + variable.name, grads[i])
                
            if global_step.numpy() % args.eval_interval == 0:
                test_AP, test_mAP = util.eval_dataset_map(model, test_dataset)
                print("mAP: ", test_mAP)
                # print("Loss: %.4f, Acc: %.4f, mAP: %.4f", test_lotest_mAP)
                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('Test mAP', test_mAP)

        if ep % 2 == 0:
            root.save(ckpt_prefix)

    model.summary()

    AP, mAP = util.eval_dataset_map(model, test_dataset)
    rand_AP = util.compute_ap(
        test_labels, np.random.random(test_labels.shape),
        test_weights, average=None)
    print('Random AP: {} mAP'.format(np.mean(rand_AP)))
    gt_AP = util.compute_ap(test_labels, test_labels, test_weights, average=None)
    print('GT AP: {} mAP'.format(np.mean(gt_AP)))
    print('Obtained {} mAP'.format(mAP))
    print('Per class:')
    for cid, cname in enumerate(CLASS_NAMES):
        print('{}: {}'.format(cname, util.get_el(AP, cid)))


if __name__ == '__main__':
    tf.enable_eager_execution()
    main()
