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


TEST_DIR = 'dataset/base/agosto/'
DATASET_DIR='dataset/npy/'
DATASET_NAME='agosto_data.npy'

IMG_SIZE = 50

def create_test_data():
    # cicla per tutte i file in train dir
    test_data = []

    for folder in tqdm(os.listdir(TEST_DIR)):
        if not folder.startswith('.'):
            
            if folder == 'empty':
                # count the difference between the number of full and empty images
                
                for img in tqdm(os.listdir(os.path.join(TEST_DIR,folder))):
                    if not img.startswith('.'):
                        # image path
                        path = os.path.join(TEST_DIR,folder,img)
                        # load in cv2 as grayscale
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        # resize image
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        # normalize the image by dividing every pixel by the variance
                        img2 = img
                        img2 = cv2.normalize(img, img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        # label the image
                        label = [1,0]
                        test_data.append([np.array(img2), np.array(label)])
                        shuffle(test_data)
                        
                    
            if folder == 'one' or folder == 'two':
                for img in tqdm(os.listdir(os.path.join(TEST_DIR,folder))):
                    if not img.startswith('.'):
                        # image path
                        path = os.path.join(TEST_DIR,folder,img)
                        # load in cv2 as grayscale
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        # resize image
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        # normalize the image by dividing every pixel by the variance
                        img2 = img
                        img2 = cv2.normalize(img, img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        # label the image
                        label = [0,1]
                        test_data.append([np.array(img2), np.array(label)])
            
    shuffle(test_data)
    np.save(os.path.join(DATASET_DIR,DATASET_NAME), test_data)
    print('test data created')
    return test_data


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

if os.path.exists(os.path.join(DATASET_DIR,DATASET_NAME)):
    test_data = np.load(os.path.join(DATASET_DIR,DATASET_NAME))
    verify(test_data)
else:
    test_data = create_test_data()



