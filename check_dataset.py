import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tflearn
import tensorflow as tf
import numpy as np
import cv2
import os
from random import shuffle
from tqdm import tqdm


DATASET_DIR='dataset/npy/'
DATASET_NAME='train_flipped_data.npy'

IMG_SIZE = 50


def verify(test_data):
    #plot the data
    fig = plt.figure()
    f = 0
    e = 0
    for num, data in enumerate(test_data[:]):
        # separate image from label
        img_data = data[0]
        img_num = data[1]
        
        if img_num[0] == 0: 
            label='full' 
            f = f + 1
            if f < 20:
                y = fig.add_subplot(4,5,f+1)
                orig = img_data
                # restore the image shape
                data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
                y.imshow(orig, cmap='gray')
                plt.title(label)
                y.axes.get_xaxis().set_visible(False)
                y.axes.get_yaxis().set_visible(False)
        else: 
            label='empty'
            e = e + 1
    
    print('\n empty images: ',e)
    print('\n full images: ',f)
    plt.show()

test_data = np.load(os.path.join(DATASET_DIR,DATASET_NAME))
verify(test_data)




