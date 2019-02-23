# Assignment 1: Object Classification with TensorFlow!

- [Visual Learning and Recognition (16-824) Spring 2019](https://sites.google.com/andrew.cmu.edu/16824-spring2019/)
- Created By: [Tao Chen](https://taochenshh.github.io/), [Rohit Girdhar](http://rohitgirdhar.github.io)
- TAs: [Rohit Girdhar](http://rohitgirdhar.github.io), [Kenny Marino](http://kennethmarino.weebly.com/), [Senthil Purushwalkam](http://www.cs.cmu.edu/~spurushw/), [Chen-Hsuan Lin](https://chenhsuanlin.bitbucket.io/), [Samantha Powers](https://www.ri.cmu.edu/ri-people/samantha-powers/), [Tao Chen](https://taochenshh.github.io/)
- Please post questions, if any, on the piazza for HW1.
- Total points: 100

In this assignment, we will learn to train multi-label image classification models using the [TensorFlow](www.tensorflow.org) (TF) framework. We will classify images from the PASCAL 2007 dataset into the objects present in the image. Your task in this assignment is to fill in the parts of code, as described in this document, perform all experiments, and submit a report with your results and analyses. You are required to use `tf.keras`, which is an official built-in high-level API. You **should** follow the code structure we defined in the steps of this assignment. Feel free to google how to do certain things if you get stuck, but put proper attribution. It is *not* acceptable to google "alexnet for PASCAL classification in tensorflow" and copy-paste that code, as that would probably not follow the code structure we define in the assignment.

In all the following tasks, coding and analysis, please write a short summary of what you tried, what worked (or didn't), and what you learned, in the report. Write the code in the files as specified. 

**Submission Requirements**:

* Please submit your report as well as your code.
* Please submit a zip file (less than 10 MB) that contains one folder named `code` which contains all your code files and one `report` in pdf format. You should name your report as `<AndrewID>.pdf`. And please zip all your files into a single file named `<AndrewID>.zip`. 
* In the begining of the report, please include a section which lists all the commands for TAs to run your code.
* You should also mention any collaborators or other sources used for different parts of the assignment.
* The submission link for homework 1 is [here](https://goo.gl/forms/vsOQJHjZulsZrCRG2). Please use your andrew email for the submission.

## Software setup

We will use the following python libraries for the homework:

1. [TensorFlow (**1.12**)](https://www.tensorflow.org/install/pip)
1. Numpy
1. Matplotlib
1. Pillow (PIL)
1. sklearn

## Task 0: Fashion MNIST classification in TensorFlow (5 points)

[Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) is a dataset of [Zalando's](https://jobs.zalando.com/tech/) article images — consisting of 70,000 grayscale images in 10 categories. Each example is a 28x28 grayscale image, associated with a label from 10 classes. `Fashion-MNIST` is intended to serve as a direct **drop-in replacement** for the original [MNIST dataset](http://yann.lecun.com/exdb/mnist/) — often used as the "Hello, World" of machine learning programs for computer vision. It shares the same image size and structure of training and testing splits. We will use 60,000 images to train the network and 10,000 images to evaluate how accurately the network learned to classify images. 

To get started, we already provide two sample code files to get you familiar with TensorFlow and its high-level API (`tf.keras`). The first example script (`00_fashion_mnist.py`) uses a static graph to build, train, and test the model. The second example script (`01_fashion_mnist.py`) uses TensorFlow's eager execution mode which uses dynamic graph. It's also recommended you go through the [keras](https://www.tensorflow.org/guide/keras) and [eager execution](https://www.tensorflow.org/guide/eager) tutorial. 

Try running both scripts and see the difference between two modes. It will start printing the loss and accuracy. Go through the code and make sure you understand the different parts of it.

#### Q 0.1: Both scripts use the same neural network model, how many trainable parameters does each layer have?

#### Q 0.2: Show the loss and accuracy curves for both scripts with the default hyperparameters.

#### Q 0.3: Why do the plots from two scripts look different? Why does the second script show smoother loss? Why are there three jumps in the training curves?

#### Q 0.4: What happens if you train the network for 10 epochs?


## Task 1: Simple CNN network for PASCAL multi-label classification (20 points)

Now that you are familiar with both the static and dynamic graph modes in TensorFlow, you can use either mode for rest of the homework. Let's try to recognize some natural images.
We start by modifying the code to read images from the [PASCAL 2007 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/). Following steps will guide you through the process.

### Data setup

We first need to download the image dataset and annotations. Use the following commands to setup the data, and lets say it is stored at location `$DATA_DIR`.

```bash
# First, cd to a location where you want to store ~0.5GB of data.
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar -xf VOCtrainval_06-Nov-2007.tar
# Also download the test data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar && tar -xf VOCtest_06-Nov-2007.tar
cd VOCdevkit/VOC2007/
export DATA_DIR=$(pwd)
```


The first step is to write a data loader which loads this PASCAL data. Since there are only about 10K images in this dataset, we can simply load all the images into CPU memory, along with the labels. The important thing to note is that PASCAL can have multiple objects present in the same image. Hence, this is a **multi-label** classification problem, and will have to be tackled slightly differently.

We provide some starter code for this task in `02_pascal.py`. You need to fill in some of the functions, as outlined next.

### 1.1: Write a data loader for PASCAL 2007.
Find the function definition for `load_pascal` in `util.py`. As the function docstring says, the function takes as input the `$DATA_DIR` and the split (`train`/`val`/`trainval`/`test`), and outputs all the images, labels and weights from the dataset. For `N` images in the split, the images should be `np.ndarray` of shape `NxHxWx3`, and labels/weights should be `Nx20`. The labels should be 1s for each object that is present in the image, and weights should be 1 for each label in the image, except those labeled as ambiguous. All other values should be 0. For simplicity, resize all images to a canonical size (eg, 256x256px).

In the following tasks, we will use data in `trainval` for training and `test` for testing.

**Hint**: The dataset contain a `ImageSets/Main/` folder, with files named `<class_name>_<split_name>.txt`. Use those files to find images that are in the different splits of the data. Look at the README to understand the structure and labeling.


### 1.2 Data Augmentation and Dataset Generation

Since we are training a model from scratch on this small dataset, it is important to perform some basic data augmentation to avoid overfitting. Add random crops and left-right flips when training, and do a center crop when testing. As for natural images, another common practice is to substract the mean values of RGB images from ImageNet dataset. The mean values for RGB images are: `[123.68, 116.78, 103.94]`. 

**Hint**: Note that you can use functions such as `tf.image.random_flip_left_right`, `tf.random_crop` etc. These functions can be applied within `tf.data.Dataset.map`. 


### 1.3: Modify the Fashion MNIST model to be suitable for multi-label classification.
Write the code for training and testing for multi-label classification in `02_pascal.py`. We will be using the same model from Fashion MNIST (bad idea, but let's give it a shot). 


### 1.4 Measure Performance
To evaluate the trained model, we will use a standard metric for multi-label evaluation - [mean average precision (mAP)](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html). Please implement the code for evaluating the model with given dataset in function `eval_dataset_map` in `util.py`. You will need to make predictions on the given dataset with the model and call `compute_ap` to get average precision.


### 1.5 Setup tensorboard
TensorFlow ships with an awesome visualization tool called [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard). It can be used to visualize training losses, network weights and other parameters. Add code in `02_pascal.py` to visualize the testing MAP and training loss in TensorBoard.

Now that you have implemented all the parts above, we can start training the model with PASCAL 2007 dataset.

#### Q 1.1 Show clear screenshots of the learning curves of testing MAP and training loss for 5 epochs (batch size=20, learning rate=0.001). Please evaluate your model to calculate the MAP on the testing dataset every 50 iterations. 



## Task 2: Lets go deeper! CaffeNet for PASCAL classification (20 points)

As you might have seen, the performance of our simple CNN mode was pretty low for PASCAL. This is expected as PASCAL is much more complex than FASHION MNIST, and we need a much beefier model to handle it. Copy over your code from `02_pascal.py` to `03_pascal_caffenet.py`, and lets implement a deep CNN.


In this task we will be constructing a variant of the [alexnet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) architecture, known as CaffeNet. If you are familiar with Caffe, a prototxt of the network is available [here](https://github.com/BVLC/caffe/blob/master/models/bvlc_reference_caffenet/train_val.prototxt). A visualization of the network is available [here](http://ethereon.github.io/netscope/#/preset/caffenet)


### 2.1 Build CaffeNet

Here is the exact model we want to build. We use the following operator notation for the architecture:

1. Convolution: A convolution with kernel size `k`, stride `s`, output channels `n`, padding `p`, is represented as `conv(k, s, n, p)`.
2. Max Pooling: A max pool operation with kernel size `k`, stride `s` as `max_pool(k, s)`.
3. Fully connected: For `n` units, `fully_connected(n)`.

```txt
ARCHITECTURE:
	-> image
	-> conv(11, 4, 96, 'VALID')
	-> relu()
	-> max_pool(3, 2)
	-> conv(5, 1, 256, 'SAME')
	-> relu()
	-> max_pool(3, 2)
	-> conv(3, 1, 384, 'SAME')
	-> relu()
	-> conv(3, 1, 384, 'SAME')
	-> relu()
	-> conv(3, 1, 256, 'SAME')
	-> relu()
	-> max_pool(3, 2)
	-> flatten()
	-> fully_connected(4096)
	-> relu()
	-> dropout(0.5)
	-> fully_connected(4096)
	-> relu()
	-> dropout(0.5)
	-> fully_connected(20)
```

### 2.2 Setup Solver Hyperparameters

Please modify your code to use the following hyperparameter settings.

1. Change the optimizer to SGD + Momentum, with momentum of 0.9.
1. Use an exponentially decaying learning rate schedule, that starts at 0.001, and decays by 0.5 every 5K iterations.
1. Use batch size 20.

### 2.3 Save the model

Please add code for saving the model periodically (save at least **30** checkpoints during training for Task 2). Please save the models for **all the remaining scripts** (Task 3 and Task 4). And for Task 3 and Task 4, you only need to save the model in the end of training.You will need these models later. 


#### Q 2.1 Show clear screenshots of testing MAP and training loss for 60 epochs. Please evaluate your model to calculate the MAP on the testing dataset every 250 iterations. 


## Task 3: Even deeper! VGG-16 for PASCAL classification (15 points)

Hopefully we all got much better accuracy with the deeper model! Since 2012, many other deeper architectures have been proposed, and [VGG-16](https://arxiv.org/abs/1409.1556) is one of the popular ones. In this task, we attempt to further improve the performance with the "very deep" VGG-16 architecture. Copy over your code from `02_pascal.py` to `04_pascal_vgg_scratch.py` and modify the code.

### 3.1: Build VGG-16
Modify the network architecture from Task 2 to implement the VGG-16 architecture (refer to the original paper). 

### 3.2: Setup TensorBoard
Add code to use tensorboard for visualizing a) Training loss, b) Learning rate, c) Histograms of gradients, d) Training images

Use the same hyperparameter settings from Task 2, and try to train the model. 

#### Q 3.1 Add screenshots of training and testing loss, testing MAP curves, learning rate, histograms of gradients and examples of training images from TensorBoard.

## Task 4: Standing on the shoulder of the giants: finetuning from ImageNet (20 points)
As we have already seen, deep networks can sometimes be hard to optimize, while other times lead to heavy overfitting on small training sets. Many approaches have been proposed to counter this, eg, [Krahenbuhl et al. (ICLR'16)](http://arxiv.org/pdf/1511.06856.pdf) and other works we have seen in un-/self-supervised learning. However, the most effective approach remains pre-training the network on large, well-labeled datasets such as ImageNet. While training on the full ImageNet data is beyond the scope of this assignment, people have already trained many popular/standard models and released them online. In this task, we will initialize the VGG model from the previous task with pre-trained ImageNet weights, and *finetune* the network for PASCAL classification. 

Link for VGG-16 pretrained model in Keras:

```bash
https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
```

Copy over your code from `02_pascal.py` to `05_pascal_vgg_finetune.py` and modify the code.

### 4.1: Load pre-trained model
Load the pre-trained weights upto fc7 layer, and initialize fc8 weights and biases from scratch. Then train the network as before. You may use funtions such as `tf.keras.utils.get_file`, `
tf.keras.models.load_weights`. Since the pretrained model might use different names for the weights, you need to figure out how to load the weights correctly.

#### Q4.1: Use similar hyperparameter setup as in the scratch case, however, let the learning rate start from 0.0001, and decay by 0.5 every 1K iterations. Show the learning curves (training and testing loss, testing MAP) for 10 epochs. Please evaluate your model to calculate the MAP on the testing dataset every 60 iterations. 

## Task 5: Analysis (20 points)

By now we should have a good idea of training networks from scratch or from pre-trained model, and the relative performance in either scenarios. Needless to say, the performance of these models is way stronger than previous non-deep architectures we used until 2012. However, final performance is not the only metric we care about. It is important to get some intuition of what these models are really learning. Lets try some standard techniques.

#### Q5.1: Conv-1 filters
Extract and compare the conv1 filters from CaffeNet in Task 2, at different stages of the training. Show at least 3 filters.

#### Q5.2: Nearest neighbors
Pick 10 images from PASCAL test set from different classes, and compute 4 nearest neighbors of those images over the test set. You should use and compare the following feature representations for the nearest neighbors:

1. pool5 features from the CaffeNet (trained from scratch)
1. fc7 features from the CaffeNet (trained from scratch)
1. pool5 features from the VGG (finetuned from ImageNet)
1. fc7 features from VGG (finetuned from ImageNet)

Show the 10 images you chose and their 4 nearest neighbors for each case.

#### Q5.3: t-SNE visualization of intermediate features
We can also visualize how the feature representations specialize for different classes. Take 1000 random images from the test set of PASCAL, and extract caffenet (scratch) `fc7` features from those images. Compute a 2D [t-SNE projection](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) of the features, and plot them with each feature color coded by the GT class of the corresponding image. If multiple objects are active in that image, compute the color as the "mean" color of the different classes active in that image. Legend the graph with the colors for each object class.

#### Q5.4: Are some classes harder?
Show the per-class performance of your caffenet (scratch) and VGG-16 (finetuned) models. Try to explain, by observing examples from the dataset, why some classes are harder or easier than the others (consider the easiest and hardest class). Do some classes see large gains due to pre-training? Can you explain why that might happen?

## Task 6 (Extra Credit): Improve the classification performance (20 points)
Many techniques have been proposed in the literature to improve classification performance for deep networks. In this section, we try to use a recently proposed technique called [*mixup*](https://arxiv.org/abs/1710.09412). The main idea is to augment the training set with linear combinations of images and labels. Read through the paper and modify your model to implement mixup. Report your performance, along with training/test curves, and comparison with baseline in the report.


## Acknowledgements
Parts of the starter code are taken from official TensorFlow tutorials. Many thanks to the original authors!
