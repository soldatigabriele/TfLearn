import tflearn
import tensorflow as tf

import numpy as np
import cv2
import os
from random import shuffle
from tqdm import tqdm

import dataset as data

#import the cnn graphs
import graphs.cnn_2layers as cnn2
import graphs.cnn_6layers as cnn6
import graphs.cnn_8layers as cnn8
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

# create
# test_data = data.process_test_data(IMG_SIZE)
# or load
test_data = np.load(os.path.join(DATASET_DIR,'test_data.npy'))

print('Data loaded')

LR = 1e-3
CNN2 = os.path.join(MODEL_DIR,'cnn-{}-{}-{}.model'.format(IMG_SIZE, LR, '2conv'))

#reset the graph
tf.reset_default_graph()
cnn2_convnet = cnn2.convnet([None,IMG_SIZE,IMG_SIZE,1], 'input', LR)
# tenserboard_dir not needed on mac or ubuntu
cnn2_model = tflearn.DNN(cnn2_convnet, tensorboard_dir='log')


#load the model
cnn2_model.load(CNN2)
print('model ',CNN2,' loaded!')

# test_data = process_test_data()
test_data = np.load(os.path.join(DATASET_DIR,'test_data.npy'))
print('test data loaded')

''' ########## PLOT DATA and SHOW PREDICTIONS ##########'''
#plot the data
# fig = plt.figure()

c = 0
w = 0
f = 0
e = 0

for num, data in enumerate(test_data[:]):
    # full [0,1] empty [1,0]
    img_data = data[0]
    img_num = data[1]

    # 3 by 4 and the numbur is the number plus 1 (because array starts from 0)
    # y = fig.add_subplot(3,4,num+1)
    orig = img_data
    # restore the image shape
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    # the [0] is the return of the prediction and is the prob of the image being a cat
    model_cnn2_out = cnn2_model.predict([data])[0]
    # model_cnn6_out = cnn6_model.predict([data])[0]
    # model_cnn8_out = cnn8_model.predict([data])[0]
    # model_alexnet_out = alexnet_model.predict([data])[0]

    if img_num[0] == 0:
        f = f + 1
    else: 
        e = e + 1
        
    if np.argmax(model_cnn2_out) == 1: 
        # print('img_num: ',img_num)
        # print('model thinks is full')
        # print('model out: ',model_cnn2_out)
        # print('model out: ',model_cnn2_out[0])
        if img_num[0] == 0:
            c = c + 1
        else:
            w = w + 1

    else: 
        if img_num[0] == 1:
            c = c + 1
        else:
            w = w + 1

print('full: ',f)
print('empty: ',e)

print('wrong: ',w)
print('right: ',c)

    # y.imshow(orig, cmap='gray')
    # string = str_label_c,' ',model_cnn2_out,' - ',str_label_a,' ',model_alexnet_out
    # print(string)
    # plt.title(string)
    # y.axes.get_xaxis().set_visible(False)
    # y.axes.get_yaxis().set_visible(False)
# plt.show()
# # write
# with open('submission-file.csv','w') as f:
#     f.write('id,label\n')
#
# with open('submission-file.csv','a') as f:
#     for data in tqdm(test_data):
#         img_num = data[1]
#         img_data = data[0]
#         orig = img_data
#         data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
#         model_out = model.predict([data])[0]
#         f.write('{},{}\n'.format(img_num, model_out[1]))




