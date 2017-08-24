import matplotlib
import matplotlib.pyplot as plt
import tflearn
import tensorflow as tf
import numpy as np
import cv2
import os
from random import shuffle
from tqdm import tqdm

TEST_DIR = 'dataset/base/test/'
DATASET_DIR='dataset/npy/'
DATASET_NAME='test_data.npy'
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
                        path = os.path.join(TEST_DIR,folder,img)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        img2 = img
                        img2 = cv2.normalize(img, img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        # label the image
                        label = [1,0]
                        test_data.append([np.array(img2), np.array(label)])
                        shuffle(test_data)
                    
            if folder == 'one' or folder == 'two':
                for img in tqdm(os.listdir(os.path.join(TEST_DIR,folder))):
                    if not img.startswith('.'):
                        path = os.path.join(TEST_DIR,folder,img)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        img2 = img
                        img2 = cv2.normalize(img, img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        # label the image
                        label = [0,1]
                        test_data.append([np.array(img2), np.array(label)])
            
    shuffle(test_data)
    np.save(os.path.join(DATASET_DIR,DATASET_NAME), test_data)
    print('test data created')
    return test_data

test_data = create_test_data()
