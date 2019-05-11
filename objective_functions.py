'''
This script contains several objective functions implemented for the BO-BOS algorithm, including
    (1) Logistic regression trained on the MNIST dataset
    (2) Convolutional neural network (CNN) trained on the CIFAR-10 dataset
    (3) Convolutional neural network (CNN) trained on the SVHN dataset
'''

from __future__ import print_function
import tensorflow as tf
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, linear_model
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from bayesian_optimization import BayesianOptimization
import pickle
import GPy
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import time
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers

from bos_function import run_BOS

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

mnist_path = "datasets/mnist/"
svhn_path = "datasets/svhn/"

kappa = 2.0 # the \kappa parameter to be used in the second criteria for early stopping


def objective_function_LR_MNIST(param, no_stop=False, incumbent=None, bo_iteration=0, stds=[], N=50, N_init_epochs=8):
    '''
        param: parameters
        no_stop: if TRUE, then the function evaluation never early-stops
        incumbent: the currently found maximum value
        bo_iteration: the BO iteration
        stds: the standard deviations corresponding to different input number of epochs; used in the second criteria for early stopping
        N: the maximum number of epochs
        N_init_epochs: the number of initial epochs used in BOS
    '''
    
    training_epochs = N
    num_init_curve=N_init_epochs
    time_BOS = -1 # the time spent in solving the BOS problem, just for reference    
    
    #### load the MNIST dataset
    loaded_data = pickle.load(open(mnist_path + "mnist_dataset.p", "rb"))
    X_train = loaded_data["X_train"]
    X_test = loaded_data["X_test"]
    Y_train = loaded_data["Y_train"]
    Y_test = loaded_data["Y_test"]
    n_ft, n_classes = X_train.shape[1], Y_train.shape[1]

    # transform the input to the real range of the hyper-parameters, to be used for model training
    parameter_range = [[20, 500], [1e-6, 1.0], [1e-3, 0.10]]
    batch_size_ = param[0]
    batch_size = int(batch_size_ * (parameter_range[0][1] - parameter_range[0][0]) + parameter_range[0][0])
    C_ = param[1]
    C = C_ * (parameter_range[1][1] - parameter_range[1][0]) + parameter_range[1][0]
    learning_rate_ = param[2]
    learning_rate = learning_rate_ * (parameter_range[2][1] - parameter_range[2][0]) + parameter_range[2][0]
    
    print("[Evaluating parameters: batch size={0}/C={1}/lr={2}]".format(batch_size, C, learning_rate))

    
    ### The tensorflow model of logistic regression is built below

    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, n_ft]) # mnist data image of shape 28*28=784
    y = tf.placeholder(tf.float32, [None, n_classes]) # 0-9 digits recognition => 10 classes

    # Set model weights
    W = tf.Variable(tf.zeros([n_ft, n_classes]))
    b = tf.Variable(tf.zeros([n_classes]))

    # Construct model
    pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

    regularizers = tf.nn.l2_loss(W)

    # Minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1) + C * regularizers)
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    neg_log_loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    
    val_epochs = []
    time_func_eval = []
    with tf.Session(config=config) as sess:
        # Run the initializer
        sess.run(init)
        # iteration over the number of epochs
        for epoch in tqdm(range(training_epochs)):
            avg_cost = 0.0
            total_batch = int(X_train.shape[0] / batch_size)

            # Loop over all batches for SGD
            for i in range(total_batch):
                batch_xs, batch_ys = X_train[(i*batch_size):((i+1)*batch_size), :], Y_train[(i*batch_size):((i+1)*batch_size), :]
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
                avg_cost += c / total_batch
            
            # calculate validation loss
#             val_log_loss = neg_log_loss.eval({x:X_test, y:Y_test})
            val_acc = accuracy.eval({x:X_test, y:Y_test})
            val_epochs.append(val_acc)

            time_func_eval.append(time.time())

            # run BOS after observing "num_init_curve" initial number of training epochs
            if (epoch+1 == num_init_curve) and (not no_stop):
                print("initial learning errors: ", 1 - np.array(val_epochs))
                time_start = time.time()
                action_regions, grid_St = run_BOS(1 - np.array(val_epochs), incumbent, training_epochs, bo_iteration)
                time_BOS = time.time() - time_start

            # start using the decision rules obtained from BOS
            if (epoch >= num_init_curve) and (not no_stop):
                state = np.sum(1 - np.array(val_epochs[num_init_curve:])) / (epoch - num_init_curve + 1)
                ind_state = np.max(np.nonzero(state > grid_St)[0])
                action_to_take = action_regions[epoch - num_init_curve, ind_state]
                
                # condition 1: if action_to_take == 2, then the optimal decision is to stop the current training
                if action_to_take == 2:
                    # condition 2: the second criteria used in the BO-BOS algorithm
                    if (kappa * stds[epoch] >= stds[-1]) or (stds == []):
                        break

    return val_epochs[-1], (epoch + 1) / training_epochs, time_BOS, val_epochs, time_func_eval


def objective_function_CNN_CIFAR_10(param, no_stop=False, incumbent=None, bo_iteration=0, stds=[], N=50, N_init_epochs=8):
    '''
        param: parameters
        no_stop: if TRUE, then the function evaluation never early-stops
        incumbent: the currently found maximum value
        bo_iteration: the BO iteration
        stds: the standard deviations corresponding to different input number of epochs; used in the second criteria for early stopping
        N: the maximum number of epochs
        N_init_epochs: the number of initial epochs used in BOS
    '''
    
    data_augmentation = True

    training_epochs = N
    num_init_curve=N_init_epochs
    time_BOS = -1 # the time spent in solving the BOS problem, just for reference    

    #### load the CIFAR-10 dataset
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    
    # transform the input to the real range of the hyper-parameters, to be used for model training
    parameter_range = [[32, 512], [1e-7, 0.1], [1e-7, 1e-3], [1e-7, 1e-3], [128, 256], [256, 512]]
    batch_size_ = param[0]
    batch_size = int(batch_size_ * (parameter_range[0][1] - parameter_range[0][0]) + parameter_range[0][0])
    learning_rate_ = param[1]
    learning_rate = learning_rate_ * (parameter_range[1][1] - parameter_range[1][0]) + parameter_range[1][0]
    learning_rate_decay_ = param[2]
    learning_rate_decay = learning_rate_decay_ * (parameter_range[2][1] - parameter_range[2][0]) + parameter_range[2][0]
    l2_regular_ = param[3]
    l2_regular = l2_regular_ * (parameter_range[3][1] - parameter_range[3][0]) + parameter_range[3][0]
    conv_filters_ = param[4]
    conv_filters = int(conv_filters_ * (parameter_range[4][1] - parameter_range[4][0]) + parameter_range[4][0])
    dense_units_ = param[5]
    dense_units = int(dense_units_ * (parameter_range[5][1] - parameter_range[5][0]) + parameter_range[5][0])

    print("[parameters: batch_size: {0}/lr: {1}/lr_decay: {2}/l2: {3}/conv_filters: {4}/dense_unit: {5}]".format(\
        batch_size, learning_rate, learning_rate_decay, l2_regular, conv_filters, dense_units))

    num_conv_layers = 3
    dropout_rate = 0.0
    kernel_size = 5
    pool_size = 3
    
    
    # build the CNN model using Keras
    model = Sequential()
    model.add(Conv2D(conv_filters, (kernel_size, kernel_size), padding='same',
                     input_shape=x_train.shape[1:], kernel_regularizer=regularizers.l2(l2_regular)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(conv_filters, (kernel_size, kernel_size), padding='same', kernel_regularizer=regularizers.l2(l2_regular)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(dropout_rate))

    if num_conv_layers >= 3:
        model.add(Conv2D(conv_filters, (kernel_size, kernel_size), padding='same', kernel_regularizer=regularizers.l2(l2_regular)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
        model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(dense_units, kernel_regularizer=regularizers.l2(l2_regular)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    opt = keras.optimizers.rmsprop(lr=learning_rate, decay=learning_rate_decay)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    time_start = time.time()

    val_epochs = []
    time_func_eval = []
    for epoch in range(training_epochs):
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=1,
                  validation_data=(x_test, y_test),
                  shuffle=True, verbose=0)
        scores = model.evaluate(x_test, y_test, verbose=0)
        val_epochs.append(scores[1])
        time_func_eval.append(time.time())

        # run BOS after observing "num_init_curve" initial number of training epochs
        if (epoch+1 == num_init_curve) and (not no_stop):
            print("initial learning errors: ", 1 - np.array(val_epochs))
            time_start = time.time()
            action_regions, grid_St = run_BOS(1 - np.array(val_epochs), incumbent, training_epochs, bo_iteration)
            time_BOS = time.time() - time_start

        # start using the decision rules obtained from BOS
        if (epoch >= num_init_curve) and (not no_stop):
            state = np.sum(1 - np.array(val_epochs[num_init_curve:])) / (epoch - num_init_curve + 1)
            ind_state = np.max(np.nonzero(state > grid_St)[0])
            action_to_take = action_regions[epoch - num_init_curve, ind_state]

            # condition 1: if action_to_take == 2, then the optimal decision is to stop the current training
            if action_to_take == 2:
                # condition 2: the second criteria used in the BO-BOS algorithm
                if (kappa * stds[epoch] >= stds[-1]) or (stds == []):
                    break

    return val_epochs[-1], (epoch + 1) / training_epochs, time_BOS, val_epochs, time_func_eval
    
    
def objective_function_CNN_SVHN(param, no_stop=False, incumbent=None, bo_iteration=0, stds=[], N=50, N_init_epochs=8):
    '''
        param: parameters
        no_stop: if TRUE, then the function evaluation never early-stops
        incumbent: the currently found maximum value
        bo_iteration: the BO iteration
        stds: the standard deviations corresponding to different input number of epochs; used in the second criteria for early stopping
        N: the maximum number of epochs
        N_init_epochs: the number of initial epochs used in BOS
    '''
    
    data_augmentation = True

    training_epochs = N
    num_init_curve=N_init_epochs
    time_BOS = -1 # the time spent in solving the BOS problem, just for reference    

    # load the svhn dataset
    train_data = loadmat(svhn_path + "train_32x32.mat")
    test_data = loadmat(svhn_path + "test_32x32.mat")
    y_train = keras.utils.to_categorical(train_data['y'][:,0])[:,1:]
    y_test = keras.utils.to_categorical(test_data['y'][:,0])[:,1:]
    x_train = np.zeros((73257, 32, 32, 3))
    for i in range(len(x_train)):
        x_train[i] = train_data['X'].T[i].T.astype('float32')/255
    x_test = np.zeros((26032, 32, 32, 3))
    for i in range(len(x_test)):
        x_test[i] = test_data['X'].T[i].T.astype('float32')/255
    
    
    # transform the input to the real range of the hyper-parameters, to be used for model training
    parameter_range = [[32, 512], [1e-7, 0.1], [1e-7, 1e-3], [1e-7, 1e-3], [128, 256], [256, 512]]
    batch_size_ = param[0]
    batch_size = int(batch_size_ * (parameter_range[0][1] - parameter_range[0][0]) + parameter_range[0][0])
    learning_rate_ = param[1]
    learning_rate = learning_rate_ * (parameter_range[1][1] - parameter_range[1][0]) + parameter_range[1][0]
    learning_rate_decay_ = param[2]
    learning_rate_decay = learning_rate_decay_ * (parameter_range[2][1] - parameter_range[2][0]) + parameter_range[2][0]
    l2_regular_ = param[3]
    l2_regular = l2_regular_ * (parameter_range[3][1] - parameter_range[3][0]) + parameter_range[3][0]
    conv_filters_ = param[4]
    conv_filters = int(conv_filters_ * (parameter_range[4][1] - parameter_range[4][0]) + parameter_range[4][0])
    dense_units_ = param[5]
    dense_units = int(dense_units_ * (parameter_range[5][1] - parameter_range[5][0]) + parameter_range[5][0])

    print("[parameters: batch_size: {0}/lr: {1}/lr_decay: {2}/l2: {3}/conv_filters: {4}/dense_unit: {5}]".format(\
        batch_size, learning_rate, learning_rate_decay, l2_regular, conv_filters, dense_units))

    num_conv_layers = 3
    dropout_rate = 0.0
    kernel_size = 5
    pool_size = 3
    
    
    # build the CNN model using Keras
    model = Sequential()
    model.add(Conv2D(conv_filters, (kernel_size, kernel_size), padding='same',
                     input_shape=x_train.shape[1:], kernel_regularizer=regularizers.l2(l2_regular)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(conv_filters, (kernel_size, kernel_size), padding='same', kernel_regularizer=regularizers.l2(l2_regular)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(dropout_rate))

    if num_conv_layers >= 3:
        model.add(Conv2D(conv_filters, (kernel_size, kernel_size), padding='same', kernel_regularizer=regularizers.l2(l2_regular)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
        model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(dense_units, kernel_regularizer=regularizers.l2(l2_regular)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    opt = keras.optimizers.rmsprop(lr=learning_rate, decay=learning_rate_decay)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    time_start = time.time()

    val_epochs = []
    time_func_eval = []
    for epoch in range(training_epochs):
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=1,
                  validation_data=(x_test, y_test),
                  shuffle=True, verbose=0)
        scores = model.evaluate(x_test, y_test, verbose=0)
        val_epochs.append(scores[1])
        time_func_eval.append(time.time())

        # run BOS after observing "num_init_curve" initial number of training epochs
        if (epoch+1 == num_init_curve) and (not no_stop):
            print("initial learning errors: ", 1 - np.array(val_epochs))
            time_start = time.time()
            action_regions, grid_St = run_BOS(1 - np.array(val_epochs), incumbent, training_epochs, bo_iteration)
            time_BOS = time.time() - time_start

        # start using the decision rules obtained from BOS
        if (epoch >= num_init_curve) and (not no_stop):
            state = np.sum(1 - np.array(val_epochs[num_init_curve:])) / (epoch - num_init_curve + 1)
            ind_state = np.max(np.nonzero(state > grid_St)[0])
            action_to_take = action_regions[epoch - num_init_curve, ind_state]

            # condition 1: if action_to_take == 2, then the optimal decision is to stop the current training
            if action_to_take == 2:
                # condition 2: the second criteria used in the BO-BOS algorithm
                if (kappa * stds[epoch] >= stds[-1]) or (stds == []):
                    break

    return val_epochs[-1], (epoch + 1) / training_epochs, time_BOS, val_epochs, time_func_eval


