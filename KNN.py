import time, random
import numpy as np
from keras.datasets import cifar10
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as matplot

# Set seed for replicable results
np.random.seed(1)
random.seed(1)

# Load dataset
(train_img, train_lbl), (test_img, test_lbl) = cifar10.load_data()

# Reshape data
train_img = np.reshape(train_img, (train_img.shape[0], -1))
test_img = np.reshape(test_img, (test_img.shape[0], -1))
train_lbl = np.reshape(train_lbl, -1)
test_lbl = np.reshape(test_lbl, -1)

# Normalize the data
train_img = train_img/255
test_img = test_img/255

# Define a subset of the training data
train_subset = 1000
test_subset = int(train_subset/10)

def data_subset(images, labels, subset_size):
    label_list = []
    for i in range(10):
        label_list += np.array(np.where(labels == i)).tolist()[0][0:subset_size]
    random.shuffle(label_list)
    return images[label_list,:], labels[label_list]

# Create a subset for train images
train_img, train_lbl = data_subset(train_img, train_lbl, train_subset)

# Create a subset for test images
test_img, test_lbl = data_subset(test_img, test_lbl, test_subset)

# KNN Method

numberNeighbors = []
times = []
accuracies = []

for n in range(1,16):
    timerStart = time.time()

    knn = KNeighborsClassifier(n_neighbors=n)

    knn.fit(train_img, train_lbl)
    lblPrediction = knn.predict(test_img)

    timerEnd = time.time()

    timeToRun = timerEnd - timerStart

    modelAccuracy = metrics.accuracy_score(test_lbl, lblPrediction)

    numberNeighbors.append(n)
    print("Number of neighbors: ", n, "\n")

    times.append(timeToRun)
    print("Time to run: ", timeToRun, "\n")

    accuracies.append(modelAccuracy)
    print("Accuracy: ", modelAccuracy, "\n")

print("Accuracies: ", accuracies)
print("Times: ", times)
matplot.xticks(numberNeighbors)
fig, axs = matplot.subplots(1, 1, figsize=(10, 10))
axs.plot(numberNeighbors, accuracies, color="red", marker="o")
# Plot settings
axs.set_ylabel("Accuracy", color="red")
axs.set(title="KNN accuracy plot", ylabel="Accuracy", xlabel="Neighbors")
axs2 = axs.twinx()
axs2.plot(numberNeighbors, times, color="blue", marker="o")
axs2.set_ylabel("Time to run", color="blue")
matplot.savefig("KNN.png")
matplot.clf()