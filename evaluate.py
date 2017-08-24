import tflearn
import tensorflow as tf
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

GRAPHS_DIR='graphs/'
#have to be in the same directory of the script or terminal
DATASET_DIR='dataset/validation/'
MODEL_DIR='models/'
FILE_NAME='summary_norm.csv'
NOTES = 'note?'

# 50x50 pixel
IMG_SIZE = 50

#create summary file
with open(FILE_NAME, 'w') as f:
    f.write('dataset,model,pp,pv,pt,vp,vv,vt,perc_full,perc_empty,perc_tot\n')



def eval(model_graph):
    print('model_graph: ',model_graph)
    tf.reset_default_graph()
    LR=model_graph.split('-')[2]
    dot_split = model_graph.split('.')
    dash_split = dot_split[-2].split('-')
    model = dash_split[-1]
    model_name = model
    graph = globals()[model]
    network = graph.network([None,IMG_SIZE,IMG_SIZE,1], 'input', LR)
    model = tflearn.DNN(network, tensorboard_dir='log')
    name = os.path.join(MODEL_DIR, model_graph)
    model.load(name)
    print('Model successfully loaded from ', name)

    # load
    for dataset in os.listdir(DATASET_DIR):
        test_data = np.load(os.path.join(DATASET_DIR,dataset))
        print('test_data:', dataset)
    
        c1 = 0
        c2 = 0
        w1 = 0
        w2 = 0
        ft = 0
        et = 0

        for num, data in enumerate(test_data[:]):
            # full [0,1] empty [1,0]
            img_data = data[0]
            img_num = data[1]

            orig = img_data
            # restore the image shape
            data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
            # the [0] is the return of the prediction and is the prob of the image being a cat
            model_out = model.predict([data])[0]

            if img_num[0] == 0: ft = ft + 1
            else: et = et + 1
                
            if np.argmax(model_out) == 1: 
                # print('model thinks is full')
                if img_num[0] == 0: c1 = c1 + 1
                else: w1 = w1 + 1

            else: 
                # print('model thinks is empty')
                if img_num[0] == 1: c2 = c2 + 1
                else: w2 = w2 + 1

        
        print('full: ',ft)
        print('empty: ',et)
        print('classificata come piena ma era vuota : ',w1)
        print('classificata come vuota ma era piena: ',w2)
        print('classificata come piena ed era piena : ',c1)
        print('classificata come vuota ed era vuota: ',c2)

        # % full right
        perc_full = c1 * 100 / ft
        perc_empty = c2 * 100 / et
        perc_tot = (c1+c2)*100/(ft+et)
        perc_full = float("{0:.2f}".format(perc_full))
        perc_empty = float("{0:.2f}".format(perc_empty))
        perc_tot = float("{0:.2f}".format(perc_tot))

        # write
        with open(FILE_NAME, 'a') as f:
            f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(dataset,model_name,c1,w2,ft,w1,c2,et,perc_full,perc_empty,perc_tot))

    

# REPEAT FOR ALL THE MODELS
for data in tqdm(os.listdir(MODEL_DIR)):
    if not data.startswith('.'):
        model_name = data.split('.')
        model_ext = model_name[-1]
        model_graph = '.'.join(model_name[:-1])
        if model_ext =='data-00000-of-00001':
            eval(model_graph)



with open(FILE_NAME, 'a') as f:
    f.write('{}\n'.format(NOTES))


    ########## PLOT DATA and SHOW PREDICTIONS ##########
    #plot the data
# fig = plt.figure()

# #reset the graph
# tf.reset_default_graph()
# conv2_convnet = conv2.convnet([None,IMG_SIZE,IMG_SIZE,1], 'input', LR)
# # tenserboard_dir not needed on mac or ubuntu
# conv2_model = tflearn.DNN(conv2_convnet, tensorboard_dir='log')
# #load the model
# conv2_model.load(CNN2)
# print('model ',CNN2,' loaded!')

# def a():
#     #reset the graph
#     tf.reset_default_graph()
#     conv2_convnet = conv2.convnet([None,IMG_SIZE,IMG_SIZE,1], 'input', LR)
#     # tenserboard_dir not needed on mac or ubuntu
#     conv2_model = tflearn.DNN(conv2_convnet, tensorboard_dir='log')
#     #load the model
#     conv2_model.load(CNN2)
#     print('model ',CNN2,' loaded!')
#
#     c = 0
#     w = 0
#     f = 0
#     e = 0
#
#     for num, data in enumerate(test_data[:]):
#         # full [0,1] empty [1,0]
#         img_data = data[0]
#         img_num = data[1]
#
#         # 3 by 4 and the numbur is the number plus 1 (because array starts from 0)
#         # y = fig.add_subplot(3,4,num+1)
#         orig = img_data
#         # restore the image shape
#         data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
#         # the [0] is the return of the prediction and is the prob of the image being a cat
#         model_out = model.predict([data])[0]
#         # model_conv6_out = conv6_model.predict([data])[0]
#         # model_conv8_out = conv8_model.predict([data])[0]
#         # model_alexnet_out = alexnet_model.predict([data])[0]
#
#         if img_num[0] == 0:
#             f = f + 1
#         else: 
#             e = e + 1
#             
#         if np.argmax(model_out) == 1: 
#             # print('img_num: ',img_num)
#             # print('model thinks is full')
#             # print('model out: ',model_conv2_out)
#             # print('model out: ',model_conv2_out[0])
#             if img_num[0] == 0:
#                 c = c + 1
#             else:
#                 w = w + 1
#
#         else: 
#             if img_num[0] == 1:
#                 c = c + 1
#             else:
#                 w = w + 1
#
#     print('full: ',f)
#     print('empty: ',e)
#
#     print('wrong: ',w)
#     print('right: ',c)
#
#         # y.imshow(orig, cmap='gray')
#         # string = str_label_c,' ',model_conv2_out,' - ',str_label_a,' ',model_alexnet_out
#         # print(string)
#         # plt.title(string)
#         # y.axes.get_xaxis().set_visible(False)
#         # y.axes.get_yaxis().set_visible(False)
#     # plt.show()
#     # # write
#     # with open('submission-file.csv','w') as f:
#     #     f.write('id,label\n')
#     #
#     # with open('submission-file.csv','a') as f:
#     #     for data in tqdm(test_data):
#     #         img_num = data[1]
#     #         img_data = data[0]
#     #         orig = img_data
#     #         data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
#     #         model_out = model.predict([data])[0]
#     #         f.write('{},{}\n'.format(img_num, model_out[1]))
#
