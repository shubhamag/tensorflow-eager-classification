import numpy as np
import sklearn.metrics
import tensorflow as tf
from tensorflow import keras
import os
import glob
import skimage.io as io
import skimage
import pdb
from tensorflow.contrib import eager as tfe

def set_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    keras.backend.set_session(session)
    return session


def set_random_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)


def load_pascal(data_dir, class_names, split='train'):
    """
    Function to read images from PASCAL data folder.
    Args:
        data_dir (str): Path to the VOC2007 directory.
        class_names (list): list of class names
        split (str): train/val/trainval split to use.
    Returns:
        images (np.ndarray): Return a np.float32 array of
            shape (N, H, W, 3), where H, W are 256px each,
            and each image is in RGB format.
        labels (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are active in that image.
        weights: (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are confidently labeled and 0s for classes that
            are ambiguous.
    """
    ## TODO Implement this function
    print("Loading Pascal data, split: ",split)
    path = os.path.join(data_dir,"ImageSets","Main")
    listpath = os.path.join(path,split + ".txt")
    fo = open(listpath, "r")
    impath = os.path.join(data_dir,"JPEGImages")
    # rl = fo.readlines()
    rl = [line.rstrip() for line in fo.readlines()]
    fo.close()
    list = [os.path.join(impath,l + ".jpg") for l in rl]
    N = len(list)
    NC = len(class_names)
    imsize = 256
    images = np.zeros([N,imsize,imsize,3],dtype=np.float32)
    labels = np.zeros([N,NC],dtype=np.int32)
    weights = np.ones_like(labels,dtype=np.int32)
    imgc=io.ImageCollection(list)
    for i,im in enumerate(imgc):
        images[i,:,:,:] = skimage.transform.resize(imgc[i], (256, 256, 3))

    print("\nDone reading and resizing images of " + split + " set.\n")
    for j,c in enumerate(class_names):
        cfile = open(os.path.join(path,c + "_" + split + ".txt"),'r')
        rfl = cfile.readlines()
        cls = [line.rstrip() for line in rfl]
        for i,cl in enumerate(cls):

            try:
                if(len(cl.split()) ==1 ):
                    labels[i][j] =0
                else:
                    cl = cl.split()[1]
                    if (int(cl)) >=0:
                        labels[i][j] = 1
                        weights[i][j] = 1
                        if (int(cl)==0): #difficult
                            weights[i][j] =0
                    else:
                        labels[i][j] = 0
            except:
                pdb.set_trace()




    ##load up images into array : skiimage ? ##



    return images, labels, weights


def cal_grad(model, loss_func, inputs, targets, weights=1.0):
    """
    Return the loss value and gradients
    Args:
         model (keras.Model): model
         loss_func: loss function to use
         inputs: image inputs
         targets: labels
         weights: weights of the samples
    Returns:
         loss and gradients
    """

    with tf.GradientTape() as tape:
        logits = model(inputs,training=True)
        loss_value = loss_func(targets, logits, weights)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def compute_ap(gt, pred, valid, average=None):
    """
    Compute the multi-label classification accuracy.
    Args:
        gt (np.ndarray): Shape Nx20, 0 or 1, 1 if the object i is present in that
            image.
        pred (np.ndarray): Shape Nx20, probability of that object in the image
            (output probablitiy).
        valid (np.ndarray): Shape Nx20, 0 if you want to ignore that class for that
            image. Some objects are labeled as ambiguous.
    Returns:
        AP (list): average precision for all classes
    """
    nclasses = gt.shape[1]
    AP = []
    for cid in range(nclasses):
        # vls = valid[:, cid] > 0
        # ids = np.where(vls)[0]
        # gtnp = gt.numpy().astype('float32')
        # prednp = pred.numpy().astype('float32')
        #
        # gt_cls = gtnp[ids,cid]
        gt_cls = gt[:, cid][valid[:, cid] > 0].astype('float32')
        pred_cls = pred[:, cid][valid[:, cid] > 0].astype('float32')
        # pred_cls = prednp[ids,cid]
        # As per PhilK. code:
        # https://github.com/philkr/voc-classification/blob/master/src/train_cls.py
        pred_cls -= 1e-5 * gt_cls
        ap = sklearn.metrics.average_precision_score(
            gt_cls, pred_cls, average=average)
        AP.append(ap)
    return AP


def eval_dataset_map(model, dataset):
    """
    Evaluate the model with the given dataset
    Args:
         model (keras.Model): model to be evaluated
         dataset (tf.data.Dataset): evaluation dataset
    Returns:
         AP (list): Average Precision for all classes
         MAP (float): mean average precision
    """
    ## TODO implement the code here
    AP = []

    prob_list =[]
    gt_list = []
    val_list = []
    # dataset = dataset.batch(1)
    for (images, labels, weights) in tfe.Iterator(dataset):
        logits = model(images,training=False)
        probs = tf.math.sigmoid(logits)
        prob_list.append(probs.numpy().reshape(-1,20))
        val_list.append(weights.numpy().reshape(-1,20))
        gt_list.append(labels.numpy().reshape(-1,20))



        # loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits)
        # prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    # pdb.set_trace()
    labels = np.concatenate(gt_list, axis=0)
    probs = np.concatenate(prob_list, axis=0)
    weights = np.concatenate(val_list, axis=0)



    # pdb.set_trace()
    AP = compute_ap(labels,probs,weights)

    mAP = sum(AP)/len(AP)

    return AP, mAP


def get_el(arr, i):
    try:
        return arr[i]
    except IndexError:
        return arr
