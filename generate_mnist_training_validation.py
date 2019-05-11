# This script is used to generate the training set/validation set split for the MNIST dataset

import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle

mnist = sio.loadmat('datasets/mnist-original.mat')

X = mnist['data'].T
Y = preprocessing.OneHotEncoder().fit_transform(mnist['label'].T).toarray()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)
scaler = preprocessing.MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

dataset = {"X_train":X_train, "Y_train":Y_train, "X_test":X_test, "Y_test":Y_test}

pickle.dump(dataset, open("datasets/mnist_dataset.p", "wb"))
