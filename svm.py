import random
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from sklearn import svm
from sklearn.model_selection import GridSearchCV
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

"""
    Subset size for parameter tuning and final model
    Subset size is per label type, which there are 10 of
"""
parameter_subset = 200
train_subset = 2000

def data_subset(images, labels, subset_size):
    label_list = []
    for i in range(10):
        label_list += np.array(np.where(labels == i)).tolist()[0][0:subset_size]
    random.shuffle(label_list)
    # scale data from 0-1
    return (images[label_list,:])/255, labels[label_list]

def parameter_tune(train_img,test_img,train_lbl,test_lbl,subset_size,c_values, gamma_values,nfolds):
    train_img, train_lbl = data_subset(train_img, train_lbl, subset_size)
    test_img, test_lbl = data_subset(test_img, test_lbl, int(subset_size/10))
    parameter_grid = dict(gamma=gamma_values,C=c_values)
    grid = GridSearchCV(svm.SVC(kernel='rbf'),parameter_grid,cv=nfolds,verbose=1)
    grid.fit(train_img,train_lbl)
    
    print(grid.best_params_)
    
    scores = grid.cv_results_['mean_test_score'].reshape(len(c_values),
                                                     len(gamma_values))
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot
               )
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_values)), gamma_values, rotation=45)
    plt.yticks(np.arange(len(c_values)), c_values)
    plt.title('Validation accuracy')
    plt.show()
    

    return grid.best_params_


# rbf
c_values = [0.001,0.01,0.1,1,10,20,50,100,1000,10000,100000,1000000]
gamma_values = [0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10]
nfolds = 2

best_params = parameter_tune(train_img, test_img, train_lbl, test_lbl, parameter_subset, c_values, gamma_values, nfolds)
c = best_params.get('C')
gamma = best_params.get('gamma')

"""
    Train better model on the best parameters
"""

# Subset for train images
train_img, train_lbl = data_subset(train_img, train_lbl, train_subset) 

# Subset for test images
test_img, test_lbl = data_subset(test_img, test_lbl, int(train_subset/10))
svmRBF = svm.SVC(kernel='rbf',cache_size=2000)
svmRBF.fit(train_img,train_lbl)
test_lbl_predicted_RBF = svmRBF.predict(test_img)
print ("RBF Accuracy: ",accuracy_score(test_lbl,test_lbl_predicted_RBF))



