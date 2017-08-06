import tflearn
import tensorflow as tf
# print(tf.__version__)
import numpy as np
import cv2
import os
from random import shuffle
from tqdm import tqdm

import dataset as data

#import the cnn graphs
import graphs.cnn_2layers as conv2
import graphs.cnn_6layers as conv6
import graphs.cnn_8layers as conv8
#import alexnet
import graphs.alexnet as alexnet
import graphs.resnext as resnext
import graphs.inception as inception
import graphs.googlenet as googlenet
import graphs.convnet as conv

#have to be in the same directory of the script or terminal
AUGM_TRAIN_DIR ='dataset/augmented/training/'
AUGM_TEST_DIR ='dataset/augmented/test/'
DATASET_DIR='dataset/npy/'
MODEL_DIR='models/'
LOG_DIR='log/'
GRAPHS_DIR='graphs/'

# 50x50 pixel
IMG_SIZE = 50
# learning rate
LR = 1e-3

#models
CNN2 = os.path.join(MODEL_DIR,'cnn-{}-{}-{}.model'.format(IMG_SIZE, LR, 'conv2'))
CNN6 = os.path.join(MODEL_DIR,'cnn-{}-{}-{}.model'.format(IMG_SIZE, LR, 'conv6'))
CNN8 = os.path.join(MODEL_DIR,'cnn-{}-{}-{}.model'.format(IMG_SIZE, LR, 'conv8'))
ALEXNET = os.path.join(MODEL_DIR,'cnn-{}-{}-{}.model'.format(IMG_SIZE, LR, 'alexnet'))
RESNEXT = os.path.join(MODEL_DIR,'cnn-{}-{}-{}.model'.format(IMG_SIZE, LR, 'resnext'))
INCEPTION = os.path.join(MODEL_DIR,'cnn-{}-{}-{}.model'.format(IMG_SIZE, LR, 'inception'))
GOOGLENET = os.path.join(MODEL_DIR,'cnn-{}-{}-{}.model'.format(IMG_SIZE, LR, 'googlenet'))
CONVNET = os.path.join(MODEL_DIR,'cnn-{}-{}-{}.model'.format(IMG_SIZE, LR, 'convnet'))


# dentro a {} mette LR, mentre nel secondo {} mette conv8
# MODEL_NAME = 'cnn-{}-{}.model'.format(IMG_SIZE, 'conv8')

# create
# train_data = data.create_train_data(IMG_SIZE)
# or load
train_data = np.load(os.path.join(DATASET_DIR,'train_norm_data.npy'))

# create
# test_data = data.process_test_data(IMG_SIZE)
# or load
test_data = np.load(os.path.join(DATASET_DIR,'test_data.npy'))

print('Data loaded')


#prende tutte le entry tranne le ultime 500
train = train_data[:-500]
# prende le ultime 500 entry per test
test = train_data[-500:]

# takes the image and reshapes it
X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# takes the labels
Y = [i[1] for i in train]

# testing accuracy
test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# takes the labels
test_y = [i[1] for i in test]

# checks for checkpoints
# if os.path.exists(os.path.join(MODEL_DIR,'{}.meta'.format(MODEL_NAME))):

''' NETWORKS '''

# ''' #### CONVNET #### '''
# # reset the graph
# tf.reset_default_graph()
#
# convnet_cnn = conv.network([None,IMG_SIZE,IMG_SIZE,1], 'input', LR)
# convnet_model = tflearn.DNN(convnet_cnn, tensorboard_verbose=3, tensorboard_dir='log')
#
# if os.path.exists(os.path.join('{}.meta'.format(CONVNET))):
#     convnet_model.load(CONVNET)
#     print('model ',CONVNET,' loaded!')
# else:
#     convnet_model.fit({'input':X},{'targets':Y}, n_epoch=100, validation_set=({'input':test_x},{'targets':test_y}), snapshot_step=5000, show_metric=True, run_id='convnet')
#     convnet_model.save(CONVNET)
#     print('model ',CONVNET,' created and saved!')
#
#
# print('models trained or loaded successfully')
#
''' #### CNN2 #### '''
#reset the graph
tf.reset_default_graph()
conv2_convnet = conv2.network([None,IMG_SIZE,IMG_SIZE,1], 'input', LR )
# tenserboard_dir not needed on mac or ubuntu
conv2_model = tflearn.DNN(conv2_convnet, tensorboard_dir='log')

if os.path.exists(os.path.join('{}.meta'.format(CNN2))):
    conv2_model.load(CNN2)
    print('model ',CNN2,' loaded!')
else:
    # TRAIN the network run_id is how you will find in tensorflow the model
    conv2_model.fit({'input':X},{'targets':Y}, n_epoch=100, validation_set=({'input':test_x},{'targets':test_y}), snapshot_step=1000, show_metric=True, run_id='conv2')
    conv2_model.save(CNN2)
    print('model ', CNN2 ,'created and saved')


''' #### CNN6 #### '''
#reset the graph
tf.reset_default_graph()
conv6_convnet = conv6.network([None,IMG_SIZE,IMG_SIZE,1], 'input', LR)
conv6_model = tflearn.DNN(conv6_convnet, tensorboard_dir='log')

if os.path.exists(os.path.join('{}.meta'.format(CNN6))):
    conv6_model.load(CNN6)
    print('model ',CNN6,' loaded!')
else:
    # TRAIN the network run_id is how you will find in tensorflow the model
    conv6_model.fit({'input':X},{'targets':Y}, n_epoch=100, validation_set=({'input':test_x},{'targets':test_y}), snapshot_step=500, show_metric=True, run_id='conv6')
    conv6_model.save(CNN6)
    print('model ', CNN6 ,'created and saved')

# ''' #### CNN8 #### '''
# #reset the graph
# tf.reset_default_graph()
# conv8_convnet = conv8.network([None,IMG_SIZE,IMG_SIZE,1], 'input', LR)
# conv8_model = tflearn.DNN(conv8_convnet, tensorboard_dir='log')
# if os.path.exists(os.path.join('{}.meta'.format(CNN8))):
#     conv8_model.load(CNN8)
#     print('model ',CNN8,' loaded!')
# else:
#     conv8_model.fit({'input':X},{'targets':Y}, n_epoch=100, validation_set=({'input':test_x},{'targets':test_y}), snapshot_step=5000, show_metric=True, run_id='conv8')
#     conv8_model.save(CNN8)
#     print('model ',CNN8,' created and saved!')
#
#
# ''' #### ALEXNET #### '''
# #reset the graph
# tf.reset_default_graph()
# alexnet_convnet = alexnet.network([None,IMG_SIZE,IMG_SIZE,1], 'input', LR)
# alexnet_model = tflearn.DNN(alexnet_convnet, tensorboard_dir='log')
#
# if os.path.exists(os.path.join('{}.meta'.format(ALEXNET))):
#     alexnet_model.load(ALEXNET)
#     print('model ',ALEXNET,' loaded!')
# else:
#     alexnet_model.fit({'input':X},{'targets':Y}, n_epoch=100, validation_set=({'input':test_x},{'targets':test_y}), snapshot_step=5000, show_metric=True, run_id='alexnet')
#     alexnet_model.save(ALEXNET)
#     print('model ',ALEXNET,' created and saved!')
#
# ''' #### RESNEXT #### '''
# #reset the graph
# tf.reset_default_graph()
# resnext_convnet = resnext.network([None,IMG_SIZE,IMG_SIZE,1], 'input', LR)
# resnext_model = tflearn.DNN(resnext_convnet, tensorboard_dir='log')
#
# if os.path.exists(os.path.join('{}.meta'.format(RESNEXT))):
#     resnext_model.load(RESNEXT)
#     print('model ',RESNEXT,' loaded!')
# else:
#     resnext_model.fit({'input':X},{'targets':Y}, n_epoch=100, validation_set=({'input':test_x},{'targets':test_y}), snapshot_step=5000, show_metric=True, run_id='resnext')
#     resnext_model.save(RESNEXT)
#     print('model ',RESNEXT,' created and saved!')
#
# ''' #### INCEPTION #### '''
# #reset the graph
# tf.reset_default_graph()
# inception_convnet = inception.network([None,IMG_SIZE,IMG_SIZE,1], 'input', LR)
# inception_model = tflearn.DNN(inception_convnet, tensorboard_dir='log')
#
# if os.path.exists(os.path.join('{}.meta'.format(INCEPTION))):
#     inception_model.load(INCEPTION)
#     print('model ',INCEPTION,' loaded!')
# else:
#     inception_model.fit({'input':X},{'targets':Y}, n_epoch=100, validation_set=({'input':test_x},{'targets':test_y}), snapshot_step=5000, show_metric=True, run_id='inception')
#     inception_model.save(INCEPTION)
#     print('model ',INCEPTION,' created and saved!')
#
#
# ''' #### GOOGLENET #### '''
# #reset the graph
# tf.reset_default_graph()
# googlenet_convnet = googlenet.network([None,IMG_SIZE,IMG_SIZE,1], 'input', LR)
# googlenet_model = tflearn.DNN(googlenet_convnet, tensorboard_dir='log')
#
# if os.path.exists(os.path.join('{}.meta'.format(GOOGLENET))):
#     googlenet_model.load(GOOGLENET)
#     print('model ',GOOGLENET,' loaded!')
# else:
#     googlenet_model.fit({'input':X},{'targets':Y}, n_epoch=100, validation_set=({'input':test_x},{'targets':test_y}), snapshot_step=5000, show_metric=True, run_id='googlenet')
#     googlenet_model.save(GOOGLENET)
#     print('model ',GOOGLENET,' created and saved!')
#

# ''' ########## TRAINING #########'''
# TRAIN the network run_id is how you will find in tensorflow the model
#model.fit({'input':X},{'targets':Y}, n_epoch=2, validation_set=({'input':test_x},{'targets':test_y}), snapshot_step=5, show_metric=True, run_id=MODEL_NAME)

''' ########## SAVE MODEL #########'''
# model.save(os.path.join(MODEL_DIR,MODEL_NAME))


# import matplotlib.pyplot as plt

# test_data = process_test_data()
test_data = np.load(os.path.join(DATASET_DIR,'test_data.npy'))
print('test data loaded')

''' ########## PLOT DATA and SHOW PREDICTIONS ##########'''
#plot the data
# fig = plt.figure()

# for num, data in enumerate(test_data[:12]):
#     # cat [1,0] dog [0,1]
#     img_num = data[1]
#     img_data = data[0]
#
#     # 3 by 4 and the numbur is the number plus 1 (because array starts from 0)
#     # y = fig.add_subplot(3,4,num+1)
#     orig = img_data
#     # restore the image shape
#     data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
#     # the [0] is the return of the prediction and is the prob of the image being a cat
#     model_conv2_out = conv2_model.predict([data])[0]
#     # model_conv6_out = conv6_model.predict([data])[0]
#     # model_conv8_out = conv8_model.predict([data])[0]
#     # model_alexnet_out = alexnet_model.predict([data])[0]
#
#     # if np.argmax(model_conv2_out) == 1: str_label_c='Dog' 
#     # else: str_label_c='Cat'
#
# #     if np.argmax(model_alexnet_out) == 1: str_label_a='Dog' 
# #     else: str_label_a='Cat'
#
#     # y.imshow(orig, cmap='gray')
#     # string = str_label_c,' ',model_conv2_out,' - ',str_label_a,' ',model_alexnet_out
#     # print(string)
#     # plt.title(string)
#     # y.axes.get_xaxis().set_visible(False)
#     # y.axes.get_yaxis().set_visible(False)
# # plt.show()
# # # write
# # with open('submission-file.csv','w') as f:
# #     f.write('id,label\n')
# #
# # with open('submission-file.csv','a') as f:
# #     for data in tqdm(test_data):
# #         img_num = data[1]
# #         img_data = data[0]
# #         orig = img_data
# #         data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
# #         model_out = model.predict([data])[0]
# #         f.write('{},{}\n'.format(img_num, model_out[1]))




