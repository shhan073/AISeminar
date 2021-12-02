"""Load from /home/USER/data/cifar10 or elsewhere; download if missing."""

import tarfile
import sys, os
import requests
# from urllib.request import urlretrieve
import pickle
#import cPickle
import numpy as np
import PIL # PIL.__version__ '7.2.0'
from PIL import Image
import cv2 # python3.8
from cv2 import *
import matplotlib.pyplot as plt

# airplanes(0), cars(1), birds(2), cats(3), deer(4)
# dogs(5), frogs(6), horses(7), ships(8), trucks(9)

def cifar10_load_data():
    r"""Return (train_images, train_labels, test_images, test_labels).

    Args:
        path (str): Directory containing CIFAR-10. Default is
            /home/USER/data/cifar10 or C:\Users\USER\data\cifar10.
            Create if nonexistant. Download CIFAR-10 if missing.

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels), each
            a matrix. Rows are examples. Columns of images are pixel values,
            with the order (red -> blue -> green). Columns of labels are a
            onehot encoding of the correct class.
    """
    url = 'https://www.cs.toronto.edu/~kriz/'
    tar = 'cifar-10-python.tar.gz'
    files = ['cifar-10-batches-py/data_batch_1',
             'cifar-10-batches-py/data_batch_2',
             'cifar-10-batches-py/data_batch_3',
             'cifar-10-batches-py/data_batch_4',
             'cifar-10-batches-py/data_batch_5',
             'cifar-10-batches-py/test_batch']

    # Set path to /home/USER/data/cifar10 or C:\Users\USER\data\cifar10
    # path = os.path.join(os.path.expanduser('~'), 'data', 'cifar10')
    path = os.path.dirname(os.path.abspath(__file__))
    print('path=', path)
    # path = the current directory

    # Create path if it doesn't exist
    # os.makedirs(path, exist_ok=True)

    
    # Download tarfile if missing
    if tar not in os.listdir(path):        
       # urlretrieve(''.join((url, tar)), os.path.join(path, tar))
        file_data = requests.get(url + tar)
        print('file_data = ', file_data)
        with open(os.path.join(path, tar), 'wb') as file:
            file.write(file_data.content)
        print("Downloaded %s to %s" % (tar, path))

    # tar_object = tarfile.open('./cifar-10-python.tar.gz')

    num_train_samples = 50000
    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')
    print('x_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)

    with tarfile.open(os.path.join(path, tar), "r:gz") as tar_object:
        members = [file for file in tar_object if file.name in files]
        members.sort(key=lambda member: member.name)
        print('members = ', members)
        for i, member in enumerate(members):
            # member =  <TarInfo 'cifar-10-batches-py/data_batch_1' at 0x2120cafa700>
            # member =  <TarInfo 'cifar-10-batches-py/data_batch_2' at 0x2120cafa580>
            # member =  <TarInfo 'cifar-10-batches-py/data_batch_3' at 0x2120cafa400>
            # member =  <TarInfo 'cifar-10-batches-py/data_batch_4' at 0x2120ca95400>
            # member =  <TarInfo 'cifar-10-batches-py/data_batch_5' at 0x2120cafa280>
            # member =  <TarInfo 'cifar-10-batches-py/test_batch' at 0x2120cafa340>
            
            fb = tar_object.extractfile(member)
            print('fb = ', fb)
            print('member = ', member)

            dict = pickle.load(fb, encoding='latin1')
            # f =  <ExFileObject name='D:\\CNN-image dimension\\cifar10\\cifar-10-python.tar.gz'>
            # type(f) =  <class 'tarfile.ExFileObject'>

            #f_array = np.frombuffer(f.read(), dtype='uint8')
            f_array = np.frombuffer(dict['data'], dtype='uint8')

            # (x_train[(i - 1) * 10000:i * 10000, :, :, :], y_train[(i - 1) * 10000:i * 10000]) = np.frombuffer(dict['data'], dtype='uint8')
         
            f_array = np.frombuffer(dict['data'], dtype='uint8')
            print('f_array.shape = ', f_array.shape)
            # f_array.shape =  (31035704,)
            # f_array.shape =  (31035320,)
            # f_array.shape =  (31035999,)
            # f_array.shape =  (31035696,)
            # f_array.shape =  (31035623,)
            # f_array.shape =  (31035526,)
  
    f = open(path +'/' + tar, 'rb')
    #dict = pickle.load(f)
    dict = pickle.load(tar_object)
    images = dict['data']
    #images = np.reshape(images, (10000, 3, 32, 32))
    labels = dict['labels']
    imagearray = np.array(images)   #   (10000, 3072)
    labelarray = np.array(labels)
    
    # Load data from tarfile
    # with tarfile.open(os.path.join(path, tar)) as tar_object:
        # Each file contains 10,000 color images and 10,000 labels
    dict = pickle.load(tar_object)
    images = dict['data']
    #images = np.reshape(images, (10000, 3, 32, 32))
    labels = dict['labels']
    imagearray = np.array(images)   #   (10000, 3072)
    labelarray = np.array(labels)


    # be careful : originally (3, 32, 32)
    train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    # train_images.shape =  (50000, 32, 32, 3)
    # test_images.shape =  (10000, 32, 32, 3)

    #path = os.path.join(os.path.expanduser('~'), 'data', 'cifar10')
    save_file = path + '/' + 'cifar10.pkl'
    if not os.path.exists(save_file):
        dataset = {}
        dataset['train_image'] = train_images
        dataset['train_label'] = train_labels
        dataset['test_image'] = test_images
        dataset['test_label'] = test_labels
        print("Creating pickle file ...")
        with open(save_file, 'wb') as f:
            pickle.dump(dataset, f, -1)
            print("Done!")

    return  (train_images, train_labels), ( test_images, test_labels)

def _onehot(integer_labels):
    """Return matrix whose rows are onehot encodings of integers."""
    n_rows = len(integer_labels)
    n_cols = integer_labels.max() + 1
    onehot = np.zeros((n_rows, n_cols), dtype='uint8')
    onehot[np.arange(n_rows), integer_labels] = 1
    return onehot

def visual_cifar10_data(x_train, t_train, x_test, t_test):

    x_train = x_train.astype(np.float32)/255.0
    x_test = x_test.astype(np.float32)/255.0
    
    t_train = _onehot(t_train)
    t_test = _onehot(t_test)

    return x_train, t_train, x_test, t_test

def image_check(image_train, t_train, image_test, t_test, check_size=5):
    
    #nchannel = image_train.shape[1]
    #rows = image_train.shape[2]
    #cols = image_train.shape[3]

    train_mask = np.random.choice(image_train.shape[0], check_size)
    train_list = train_mask.tolist()
    test_mask = np.random.choice(image_test.shape[0], check_size)
    test_list = test_mask.tolist()

    for iplot in train_list:
        ilabel = t_train[iplot]
        list_ilabel = ilabel.tolist()
        mdex = list_ilabel.index(1)
        plt.title('label = ' + str(mdex))
        #img_out = image_train[iplot].reshape(rows, cols, nchannel)
        #img_out = image_train[iplot].transpose(1, 2, 0)
        img_out = image_train[iplot]
        plt.imshow(img_out)
        plt.show()

    for iplot in test_list:
        ilabel = t_test[iplot]
        list_ilabel = ilabel.tolist()
        mdex = list_ilabel.index(1)
        plt.title('label = ' + str(mdex))
        #img_out = image_test[iplot].reshape(rows, cols, nchannel)
        #img_out = image_test[iplot].transpose(1, 2, 0)
        img_out = image_test[iplot]
        plt.imshow(img_out)
        plt.show()

(x_train, t_train), (x_test, t_test) = cifar10_load_data()
x_train, t_train, x_test, t_test = visual_cifar10_data(x_train, t_train, x_test, t_test)
# for image plot
# t_train.shape =  (50000, 10)
# t_test.shape =  (10000, 10)

image_train = x_train[1]
plt.imshow(image_train)
plt.show()

image_check(x_train, t_train, x_test, t_test, check_size=1)

if __name__ == '__main__':
    cifar10_load_data()

