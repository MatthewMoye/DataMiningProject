"""
    Need to install the following python packages
    sklearn and numpy for doing SVM, KNN, and data
    scaling with command
        pip install numpy sklearn
    keras and tensorflow to load cifar-10 data
    with command
        pip install keras tensorflow
"""

import time, random
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from sklearn import svm
from sklearn.metrics import accuracy_score


# Set seed for replicable results
np.random.seed(1)
random.seed(1)

"""
    Load dataset using Keras because its really simple.

    The train dataset has 50,000 images
    with 50,000 corresponding labels.

    The test dataset has 10,000 images.
    with 10,000 corresponding labels.

    A single image has size (32,32,3).
    This means its 32x32 pixels and is an RGB image.
    Each layer in the third dimension corresponds to 
    a Red, Green, or Blue color.
    Each element takes a value between [0,255]
"""
(train_img, train_lbl), (test_img, test_lbl) = cifar10.load_data()

"""
    Here we reshape each image from (32,32,3) to (3072).
    Labels were reshaped to remove 2nd dimension
    (50000,1) and (10000,1) changed to (50000,) (10000,)
"""
train_img = np.reshape(train_img, (train_img.shape[0], -1))
test_img = np.reshape(test_img, (test_img.shape[0], -1))
train_lbl = np.reshape(train_lbl, -1)
test_lbl = np.reshape(test_lbl, -1)


train_subset = 1000
test_subset = int(train_subset/10)

def data_subset(images, labels, subset_size):
    label_list = []
    for i in range(10):
        label_list += np.array(np.where(labels == i)).tolist()[0][0:subset_size]
    random.shuffle(label_list)
    return images[label_list,:], labels[label_list]

# Subset for train images
train_img, train_lbl = data_subset(train_img, train_lbl, train_subset) 

# Subset for test images
test_img, test_lbl = data_subset(test_img, test_lbl, test_subset)

# scale data from 0-1
train_img = train_img/255
test_img = test_img/255

# simple test of each kernerl

# Linear
svmLinear = svm.SVC(kernel='linear',cache_size=2000)
svmLinear.fit(train_img,train_lbl)
test_lbl_predicted_linear = svmLinear.predict(test_img)
print ("Linear Accuracy: ",accuracy_score(test_lbl,test_lbl_predicted_linear))

# polynomial
svmPolynomial = svm.SVC(kernel='poly',cache_size=2000)
svmPolynomial.fit(train_img,train_lbl)
test_lbl_predicted_polynomial = svmPolynomial.predict(test_img)
print ("Polynomial Accuracy: ",accuracy_score(test_lbl,test_lbl_predicted_polynomial))

# rbf
svmRBF = svm.SVC(kernel='rbf',cache_size=2000)
svmRBF.fit(train_img,train_lbl)
test_lbl_predicted_RBF = svmRBF.predict(test_img)
print ("RBF Accuracy: ",accuracy_score(test_lbl,test_lbl_predicted_RBF))

