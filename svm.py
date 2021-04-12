"""
    Python packages required:
        Tensorflow and Keras (to easily import CIFAR-10)
        matplotlib and sklearn
"""
import random, time
import numpy as np
from keras.datasets import cifar10
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Set seed for replicable results
random.seed(1)

(train_img, train_lbl), (test_img, test_lbl) = cifar10.load_data()


#Here we reshape each image from (32,32,3) to (3072).
train_img = np.reshape(train_img, (train_img.shape[0], -1))
test_img = np.reshape(test_img, (test_img.shape[0], -1))
train_lbl = np.reshape(train_lbl, -1)
test_lbl = np.reshape(test_lbl, -1)

# Get subset of dataset passed in that has ten class labels
def data_subset(images, labels, subset_size):
    label_list = []
    for i in range(10):
        label_list += np.array(np.where(labels == i)).tolist()[0][0:subset_size]
    random.shuffle(label_list)
    # scale data from 0-1
    return (images[label_list,:])/255, labels[label_list]

# Find best C and gamma value
def parameter_tune(train_img,test_img,train_lbl,test_lbl,subset_size,c_values, gamma_values):
    train_img, train_lbl = data_subset(train_img, train_lbl, subset_size)
    test_img, test_lbl = data_subset(test_img, test_lbl, int(subset_size/5))
    parameter_grid = dict(gamma=gamma_values,C=c_values)
    grid = GridSearchCV(svm.SVC(kernel='rbf'),parameter_grid,verbose=1,cv=2,n_jobs=8)
    grid.fit(train_img,train_lbl)
    
    print(grid.best_params_)
    
    scores = grid.cv_results_['mean_test_score'].reshape(len(c_values),
                                                     len(gamma_values))
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_values)), gamma_values, rotation=45)
    plt.yticks(np.arange(len(c_values)), c_values)
    plt.title('Validation accuracy')
    plt.savefig("output/SVM.png")
    
    return grid.best_params_


# Best c and gamma values according parameter_tune fuction
c = 1
gamma = 0.01

# To get c and gamma again change the find_best_params to True
find_best_params = False


if find_best_params == True:
    # Subset size for parameter tuning (10k train, 2k test)
    parameter_subset = 1000
    # C and gamma values to test on
    c_values = [0.01,0.1,1,10,20,50,100,1000,10000,100000,1000000]
    gamma_values = [0.0000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1]

    best_params = parameter_tune(train_img, test_img, train_lbl, test_lbl, parameter_subset, c_values, gamma_values)
    c = best_params.get('C')
    gamma = best_params.get('gamma')

# Run best c and gamma on full dataset and get results
start = time.time()
svmRBF = svm.SVC(kernel='rbf',cache_size=2000)
svmRBF.fit(train_img,train_lbl)
test_lbl_predicted_RBF = svmRBF.predict(test_img)
print(time.time()-start)
print ("RBF Accuracy: ",accuracy_score(test_lbl,test_lbl_predicted_RBF))
