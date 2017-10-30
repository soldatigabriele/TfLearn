from shutil import copyfile
import os, sys
from tqdm import tqdm
import cv2
import numpy as np
import base64
from random import shuffle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

li = []

TRAIN_DIR='dataset/base/divise/empty2'

for img in tqdm(os.listdir(TRAIN_DIR)):
    if not img.startswith('.'):
        print(img)
        li.append(img)
        shuffle(li)

n = len(li)-1600

for i in range(n):
    del li[0]


SAVE_DIR='dataset/base/divise/empty'

for i in range(len(li)):
    print(li[i])
    copyfile(os.path.join(TRAIN_DIR,li[i]),os.path.join(SAVE_DIR,li[i]))



# with tf.Session() as sess:
#     # Feed the image_data as input to the graph and get first prediction
#     softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
#     for folder in tqdm(os.listdir(TRAIN_DIR)):
#         if not folder.startswith('.'):
#             for img in tqdm(os.listdir(os.path.join(TRAIN_DIR,folder))):
#                 if not img.startswith('.'):
#                     path = os.path.join(TRAIN_DIR,folder,img)
#                     image = tf.gfile.FastGFile(path, 'rb').read()
#                     predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image})                    
#                     print(path)

#                     if (predictions[0][0] >= 0.5):
#                         print('vuota')
#                         fol = os.path.join(SAVE_DIR,folder,'empty/')
#                         print(fol)
#                         print(img)
#                         if not os.path.exists(fol):
#                             os.makedirs(fol)
#                         else:
#                             copyfile(path, os.path.join(fol,img))
#                     else:
#                         print('piena')
#                         print(fol)
#                         print(img)
#                         fol = os.path.join(SAVE_DIR,folder,'one/')
#                         if not os.path.exists(fol):
#                             os.makedirs(fol)
#                         else:
#                             copyfile(path, os.path.join(fol,img))