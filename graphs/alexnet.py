
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
# import tflearn.datasets.oxflower17 as oxflower17
# X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))

def network(img_shape, name, LR):

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

    # Building 'AlexNet'
    network = input_data(shape=img_shape, name=name, data_preprocessing=img_prep, data_augmentation=img_aug  )
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=LR, name='targets')
    return network

# Training
# model = tflearn.DNN(network, checkpoint_path='model_alexnet',
#                     max_checkpoints=1, tensorboard_verbose=2)
# model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,
#           show_metric=True, batch_size=64, snapshot_step=200,
#           snapshot_epoch=False, run_id='alexnet_oxflowers17')
