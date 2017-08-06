# 8 layers cnn

import tflearn
import tensorflow as tf
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

def network(img_shape, name, LR):
    
    network = input_data(shape=img_shape, name=name )

    network = conv_2d(network, 32, 2, activation='relu')
    network = max_pool_2d(network,2)

    network = conv_2d(network, 64, 2, activation='relu')
    network = max_pool_2d(network,2)

    network = conv_2d(network, 32, 2, activation='relu')
    network = max_pool_2d(network,2)

    network = conv_2d(network, 64, 2, activation='relu')
    network = max_pool_2d(network,2)

    network = conv_2d(network, 32, 2, activation='relu')
    network = max_pool_2d(network,2)

    network = conv_2d(network, 64, 2, activation='relu')
    network = max_pool_2d(network,2)

    network = fully_connected(network, 1024, activation='relu')
    network = dropout(network, 0.8)
    # 2 is the number of classes
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    
    return network
