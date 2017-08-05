# 2 layers cnn

import tflearn
import tensorflow as tf
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

def network(img_shape, name, LR):
    # # Real-time data preprocessing
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()
    #
    # # Real-time data augmentation
    img_aug = ImageAugmentation()
    img_aug.add_random_blur (sigma_max=3.0)
    img_aug.add_random_flip_leftright()
    img_aug.add_random_flip_updown()
    img_aug.add_random_90degrees_rotation(rotations=[0, 2])    
    
    network = input_data(shape=img_shape, name=name, data_preprocessing=img_prep, data_augmentation=img_aug  )

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

