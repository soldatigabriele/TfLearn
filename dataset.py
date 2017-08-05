import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tflearn
import tensorflow as tf
import numpy as np
import cv2
import os
from random import shuffle
from tqdm import tqdm



TRAIN_DIR ='dataset/base/training/'
# TRAIN_DIR ='dataset/base/trying/'
TEST_DIR = 'dataset/base/test/'
DATASET_DIR='dataset/npy/'

IMG_SIZE = 50

# # 2D grayscale data
# def label_img(img):
#     #split the name by . and take the third negative (dog or cat):
#     # dog . 93 . png 
#     # [-3] [-2] [-1]
#     word_label = img.split('.')[-3]
#     # praticamente vado a creare le label in formato one_hot
#     if word_label == 'cat': return [1,0]
#     elif word_label == 'dog':return[0,1]

def norm_image(img):
    """
    Normalize PIL image
    
    Normalizes luminance to (mean,std)=(0,1), and applies a [1%, 99%] contrast stretch
    """
    img_y, img_b, img_r = img.convert('YCbCr').split()
    
    img_y_np = np.asarray(img_y).astype(float)

    img_y_np /= 255
    img_y_np -= img_y_np.mean()
    img_y_np /= img_y_np.std()
    scale = np.max([np.abs(np.percentile(img_y_np, 1.0)),
                    np.abs(np.percentile(img_y_np, 99.0))])
    img_y_np = img_y_np / scale
    img_y_np = np.clip(img_y_np, -1.0, 1.0)
    img_y_np = (img_y_np + 1.0) / 2.0
    
    img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)

    img_y = Image.fromarray(img_y_np)

    img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))
    
    img_nrm = img_ybr.convert('RGB')
    
    return img_nrm



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
    np.save(os.path.join(DATASET_DIR,'train_norm_data.npy'), training_data)
    return training_data

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
    np.save(os.path.join(DATASET_DIR,'test_data.npy'), test_data)
    print('test data created')
    return test_data

#plot the data
# fig = plt.figure()

def verify(test_data):
    f = 0
    e = 0
    for num, data in enumerate(test_data[:]):
        # separate image from label
        img_data = data[0]
        img_num = data[1]
        # print(img_data.shape)
        # 3 by 4 and the numbur is the number plus 1 (because array starts from 0)
        # if num < 16:
        #     y = fig.add_subplot(4,4,num+1)
        orig = img_data
        # restore the image shape
        data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
        # the [0] is the return of the prediction and is the prob of the image being a cat
        
        if img_num[0] == 0: 
            label='full' 
            f = f + 1
        else: 
            label='empty'
            e = e + 1
        
        # if num < 16:
    #         y.imshow(orig, cmap='gray')
    #         plt.title(label)
    #         y.axes.get_xaxis().set_visible(False)
    #         y.axes.get_yaxis().set_visible(False)
    # plt.show()
    print('\n empty images: ',e)
    print('\n full images: ',f)


# create the training normalized data
# train_data = create_train_data()
# print('data created')

# train_data = np.load('dataset/npy/train_norm_data.npy')
# train_data = np.load('dataset/npy/test_data.npy')
# print('data loaded')
# (e,i)=count_full_images()
# verify(train_data)

# if data is already present
# train_data = np.load(os.path.join(DATASET_DIR,'train_data.npy'))

# test_data = create_test_data()
# test_data = np.load(os.path.join(DATASET_DIR,'test_data.npy'))


