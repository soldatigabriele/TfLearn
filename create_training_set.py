import tflearn
import tensorflow as tf
import numpy as np
import cv2
import os
from random import shuffle
from tqdm import tqdm

TRAIN_DIR ='dataset/base/training/'
# TRAIN_DIR ='dataset/base/trying/'
DATASET_DIR='dataset/npy/'
DATASET_NAME='training_data.npy'
IMG_SIZE = 50

def count_full_images():
    # cicla per tutte i file in train dir
    i = 0
    e = 0
    for folder in tqdm(os.listdir(TRAIN_DIR)):
        if not folder.startswith('.'):
            for img in tqdm(os.listdir(os.path.join(TRAIN_DIR,folder))):
                if not img.startswith('.'):
                    if folder == 'empty':
                        label = [1,0]
                        e = e + 1
                    elif folder == 'one':
                        label = [0,1]
                        i = i + 1
                    elif folder == 'two':
                        label = [0,1]
                        i = i + 1
    return e, i


def create_train_data():
    # cicla per tutte i file in train dir
    training_data = []
    (empty_num,full_num)=count_full_images()
    for folder in tqdm(os.listdir(TRAIN_DIR)):
        if not folder.startswith('.'):
            if folder == 'empty':
                # count the difference between the number of full and empty images
                num = empty_num - full_num
                for img in tqdm(os.listdir(os.path.join(TRAIN_DIR,folder))):
                    if not img.startswith('.'):
                        # image path
                        path = os.path.join(TRAIN_DIR,folder,img)
                        # load in cv2 as grayscale
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        # resize image
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        # normalize the image by dividing every pixel by the variance
                        img2 = img
                        img2 = cv2.normalize(img, img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        # label the image
                        label = [1,0]
                        training_data.append([np.array(img2), np.array(label)])
                        shuffle(training_data)
                # remove the elements from the list to have the same number of full and empty drops
                for _ in range(num):
                    del training_data[0]
                
                print('difference: ',num)
                print('len training data: ',len(training_data))
                print('empty folder DONE')
                    
            print('len training data: ',len(training_data))
            if folder == 'one' or folder == 'two':
                for img in tqdm(os.listdir(os.path.join(TRAIN_DIR,folder))):
                    if not img.startswith('.'):
                        # image path
                        path = os.path.join(TRAIN_DIR,folder,img)
                        # load in cv2 as grayscale
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        # resize image
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        # normalize the image by dividing every pixel by the variance
                        img2 = img
                        img2 = cv2.normalize(img, img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        # label the image
                        label = [0,1]
                        training_data.append([np.array(img2), np.array(label)])
                print(folder,' DONE')
            
    shuffle(training_data)
    print('train_data length: ',len(training_data))
    np.save(os.path.join(DATASET_DIR,DATASET_NAME), training_data)
    return training_data


# create the training normalized data
train_data = create_train_data()
print('data created')

