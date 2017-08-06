# -*- coding: utf-8 -*-
""" Aggregated Residual Transformations for Deep Neural Network.
Applying a 'ResNeXT' to CIFAR-10 Dataset classification task.
References:
    - S. Xie, R. Girshick, P. Dollar, Z. Tu and K. He. Aggregated Residual
        Transformations for Deep Neural Networks, 2016.
Links:
    - [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf)
    - [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
"""

from __future__ import division, print_function, absolute_import

import tflearn

from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
# Residual blocks
# 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
n = 5

# Data loading
# from tflearn.datasets import cifar10
# (X, Y), (testX, testY) = cifar10.load_data()
# Y = tflearn.data_utils.to_categorical(Y, 10)
# testY = tflearn.data_utils.to_categorical(testY, 10)
#
# # Real-time data preprocessing
# img_prep = tflearn.ImagePreprocessing()
# img_prep.add_featurewise_zero_center(per_channel=True)
#
# # Real-time data augmentation
# img_aug = tflearn.ImageAugmentation()
# img_aug.add_random_flip_leftright()
# img_aug.add_random_crop([32, 32], padding=4)

def network(img_shape, name, LR):

    # Building Residual Network
    network = tflearn.input_data(shape=img_shape, name=name )
    network = tflearn.conv_2d(network, 16, 3, regularizer='L2', weight_decay=0.0001)
    network = tflearn.resnext_block(network, n, 16, 32)
    network = tflearn.resnext_block(network, 1, 32, 32, downsample=True)
    network = tflearn.resnext_block(network, n-1, 32, 32)
    network = tflearn.resnext_block(network, 1, 64, 32, downsample=True)
    network = tflearn.resnext_block(network, n-1, 64, 32)
    network = tflearn.batch_normalization(network)
    network = tflearn.activation(network, 'relu')
    network = tflearn.global_avg_pool(network)
    # Regression
    network = tflearn.fully_connected(network, 2, activation='softmax')
    opt = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
    network = tflearn.regression(network, optimizer=opt, name='targets', loss='categorical_crossentropy')
    return network

# Training
# model = tflearn.DNN(network, checkpoint_path='model_resnext_cifar10',
#                     max_checkpoints=10, tensorboard_verbose=0,
#                     clip_gradients=0.)
#
# model.fit(X, Y, n_epoch=200, validation_set=(testX, testY),
#           snapshot_epoch=False, snapshot_step=500,
#           show_metric=True, batch_size=128, shuffle=True,
#           run_id='resnext_cifar10')
