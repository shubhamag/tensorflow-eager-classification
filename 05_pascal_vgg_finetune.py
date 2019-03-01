from __future__ import absolute_import, division, print_function

import argparse
import os
import shutil
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pdb
from tensorflow.contrib import eager as tfe
# from tensorflow.contrib.eager import Iterator
import util

CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

class SimpleCNN(keras.Model):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__(name='VGG-16')
        self.num_classes = num_classes
        self.conv1= layers.Conv2D(64, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block1_conv1')
        self.conv2 = layers.Conv2D(64, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block1_conv2')
        self.pool1 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')

        # Block 2
        self.conv3= layers.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block2_conv1')
        self.conv4 = layers.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block2_conv2')
        self.pool2= layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')

        # Block 3
        self.conv5 = layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv1')
        self.conv6 = layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv2')
        self.conv7 = layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv3')
        self.pool3 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')

        # Block 4
        self.conv8 = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv1')
        self.conv9= layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv2')
        self.conv10= layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv3')
        self.pool4 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')

        # Block 5
        self.conv11 = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv1')
        self.conv12 = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv2')
        self.conv13 = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv3')
        self.pool5 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')

        self.flat = layers.Flatten()

        self.dense1 = layers.Dense(4096, activation='relu', name='fc1')
        self.dropout = layers.Dropout(rate=0.5)
        self.dense2 = layers.Dense(4096, activation='relu', name='fc2')
        self.dropout2 = layers.Dropout(rate=0.5)
        self.dense3 = layers.Dense(num_classes)

    def call(self, inputs, training=False):

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)

        flat_x = self.flat(x)
        out = self.dense1(flat_x)
        out = self.dropout(out, training=training)
        out = self.dense2(out)
        out = self.dropout2(out, training=training)
        out = self.dense3(out)
        return out

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape = [shape[0], self.num_classes]
        return tf.TensorShape(shape)

def flip(img, lbl,wts):
    image = tf.image.flip_left_right(img)
    return image, lbl,wts

def crop (img,lbl,wts):
    image = tf.image.random_crop(img,[224,224,3])

    return image,lbl,wts

def center_crop (img,lbl,wts):
    # image = tf.image.central_crop(img,[224,224,3])
    image = tf.image.resize_image_with_crop_or_pad(
                                    img,
                                    224,
                                    224)
    return image, lbl, wts


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
    parser.add_argument('--log-dir', type=str, default='04_vgg_pretrained_tb',
                        help='path for logging directory')
    parser.add_argument('--data-dir', type=str, default='./VOCdevkit/VOC2007',
                        help='Path to PASCAL data storage')
    args = parser.parse_args()
    util.set_random_seed(args.seed)
    sess = util.set_session()

    splt = "trainval"
    trainval_npz = splt + '.npz'
    test_npz = 'test.npz'

    if (os.path.isfile(trainval_npz)):
        print("\nFound trainval npz file\n")
        with np.load(trainval_npz) as tr_npzfile:
            train_images = tr_npzfile['imgs']
            train_labels = tr_npzfile['labels']
            train_weights = tr_npzfile['weights']
    else:

        train_images, train_labels, train_weights = util.load_pascal(args.data_dir,
                                                                     class_names=CLASS_NAMES,
                                                                     split=splt)
        np.savez(trainval_npz,imgs = train_images,labels = train_labels,weights = train_weights)

    ##TEST##
    if (os.path.isfile(test_npz)):
        print("\nFound test npz file\n")
        # npzfile = np.load(test_npz)
        with np.load(test_npz) as test_npzfile:
            test_images = test_npzfile['imgs']
            test_labels = test_npzfile['labels']
            test_weights = test_npzfile['weights']
    else:
        test_images, test_labels, test_weights = util.load_pascal(args.data_dir,
                                                                  class_names=CLASS_NAMES,
                                                                  split='test')
        np.savez(test_npz, imgs=test_images, labels=test_labels, weights=test_weights)


    ## TODO modify the following code to apply data augmentation here
    rgb_mean = np.array([123.68, 116.78, 103.94],dtype=np.float32) / 256.0
    train_images = (train_images - rgb_mean).astype(np.float32)
    test_images = (test_images - rgb_mean).astype(np.float32)

    flip_fn = lambda img,lbl,wts: flip(img, lbl,wts)
    crop_fn = lambda img,lbl,wts: crop(img, lbl,wts)
    ccrop_fn = lambda img,lbl,wts : center_crop(img,lbl,wts)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels, train_weights))
    flipped_train = train_dataset.map(flip_fn,num_parallel_calls=4)
    train_dataset = train_dataset.concatenate(flipped_train)
    train_dataset = train_dataset.map(crop_fn,num_parallel_calls=4)


    train_dataset = train_dataset.shuffle(10000).batch(args.batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels, test_weights))
    test_dataset = test_dataset.map(ccrop_fn,num_parallel_calls=4)
    test_dataset = test_dataset.batch(args.batch_size)

    model = SimpleCNN(num_classes=len(CLASS_NAMES))

    logdir = os.path.join(args.log_dir,
                          datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)

    writer = tf.contrib.summary.create_file_writer(logdir)
    writer.set_as_default()
    tf.contrib.summary.initialize()

    global_step = tf.train.get_or_create_global_step()
    # optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)

    ##decay lr using callback
    learning_rate=tf.Variable(args.lr)
    decay_interval = 5000
    # decay_op = tf.train.exponential_decay(args.lr,global_step,decay_interval,0.5)
    ##optimizer : sgd , momentum, 0.9
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    train_log = {'iter': [], 'loss': []}
    test_log = {'iter': [], 'mAP': []}
    checkpoint_directory = "./05_vgg_pretrained/"
    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    # pdb.set_trace()
    latest = tf.train.latest_checkpoint(checkpoint_directory)
    load_flag = 0
    if (latest is not None):
        print("Loading checkpoint ",latest)
        status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
        load_flag =1

    weight_load_flag = 0


    print("\nUsing eval interval: ",args.eval_interval)
    print("\nUsing batch size: ",args.batch_size)
    for ep in range(args.epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        # for batch, (images, labels,weights) in enumerate(train_dataset):
        for (images, labels,weights) in tfe.Iterator(train_dataset):
            # pdb.set_trace()
            # loss_value, grads = util.cal_grad(model,
            #                                   loss_func=tf.losses.sigmoid_cross_entropy,
            #                                   inputs=images,
            #                                   targets=labels,
            #                                   weights=weights)
            if(weight_load_flag==0):
                logits = model(images, training=True)
                model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5',by_name=True   )
                weight_load_flag =1

            with tf.GradientTape() as tape:
                logits = model(images,training=True)
                loss_value = tf.losses.sigmoid_cross_entropy(labels, logits, weights)
            grads =  tape.gradient(loss_value, model.trainable_variables)

            # print("Loss and gradient calculation, done \n")
            # pdb.set_trace()

            optimizer.apply_gradients(zip(grads,
                                          model.trainable_variables),
                                          global_step)
            epoch_loss_avg(loss_value)

            if global_step.numpy() % args.log_interval == 0:
                # pdb.set_trace()

                print('Epoch: {0:d}/{1:d} Iteration:{2:d}  Training Loss:{3:.4f}  '.format(ep,
                                                         args.epochs,
                                                         global_step.numpy(),
                                                         epoch_loss_avg.result()))
                train_log['iter'].append(global_step.numpy())
                train_log['loss'].append(epoch_loss_avg.result())

                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('Training loss', loss_value)
                    tf.contrib.summary.image('Training images', images)
                    tf.contrib.summary.scalar('Learning rate', learning_rate)
                    for i, variable in enumerate(model.trainable_variables):
                        tf.contrib.summary.histogram("grad_" + variable.name, grads[i])


            if global_step.numpy() % args.eval_interval == 0:
                print("\n **** Running Eval *****\n")
                test_AP, test_mAP = util.eval_dataset_map(model, test_dataset)
                print("Eval finsished with test mAP : ",test_mAP)
                test_log['iter'].append(global_step.numpy())
                test_log['mAP'].append(test_mAP)
                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('Testing mAP', test_mAP)

        learning_rate.assign(tf.train.exponential_decay(args.lr, global_step, decay_interval, 0.5)())
        print("Learning rate:", learning_rate)
        checkpoint.save(checkpoint_prefix)




    ## TODO write the training and testing code for multi-label classification

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
