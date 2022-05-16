"""
Image classification using a Neural Network model
"""
# Import relevant packages
 # base tools
import sys,os
import datetime
 # argument parser
import argparse
 # image processing
import cv2
 # machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_openml
 # neural networks with numpy
import numpy as np
sys.path.append(os.path.join("."))
from utils.neuralnetwork import NeuralNetwork
 # tools from tensorflow
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

""" Basic functions """
# Min-max normalisation function
def minmax(data):
    X_norm = (data-data.min())/(data.max()-data.min())
    return X_norm    

# Function to save classification report as TXT
def report_to_txt(report, name, model):
    outpath = os.path.join("out", "nn", f"report_{name}_{model}.txt")
    with open(outpath,"w") as file:
        file.write(str(report))

# Argument parser
def parse_args():
    ap = argparse.ArgumentParser()
    # dataset argument (choose between MNIST_784 and CIFAR_10)
    ap.add_argument("-d", 
                    "--dataset", 
                    required = True, 
                    help = "The dataset to train your model with; MNIST_784 or CIFAR_10")
    # model argument (choose between numpy and tensorflow)
    ap.add_argument("-m",
                    "--model",
                    required = True,
                    help = "The neural network model you wish to train on your data; numpy or tensorflow")
    args = vars(ap.parse_args())
    return args 


"""" Data loading and splitting functions """
# For the MNIST_784 dataset
def MNIST_784():
    # fetch MNIST_784 from OpenML
    X, y = fetch_openml("mnist_784", return_X_y = True)
    # get correct labels
    labels = sorted(set(y))
    # convert data into numpy arrays
    X = np.array(X)
    y = np.array(y)
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state = 42,
                                                        train_size = 7500,
                                                        test_size = 2500)
    # min-max normalisation
    X_train_scaled = minmax(X_train)
    X_test_scaled = minmax(X_test)
    # binarise labels
    y_train_lb = LabelBinarizer().fit_transform(y_train)
    y_test_lb = LabelBinarizer().fit_transform(y_test)
    return X_train_scaled, X_test_scaled, y_train_lb, y_test_lb, labels
 
# For the CIFAR_10 dataset
def CIFAR_10():
    # load the data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # get labels
    labels = ["airplane",
              "automobile",
              "bird",
              "cat",
              "deer",
              "dog",
              "frog",
              "horse",
              "ship",
              "truck"]
    # convert data to greyscale and into numpy arrays
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])
    # min-max normalisation
    X_train_scaled = minmax(X_train_grey)
    X_test_scaled = minmax(X_test_grey)
    # reshape the data
    train_nsamples, train_nx, train_ny = X_train_scaled.shape
    X_train_dataset = X_train_scaled.reshape((train_nsamples,train_nx*train_ny))
    test_nsamples, test_nx, test_ny = X_test_scaled.shape
    X_test_dataset = X_test_scaled.reshape((test_nsamples,test_nx*test_ny))
    # binarise labels
    y_train_lb = LabelBinarizer().fit_transform(y_train)
    y_test_lb = LabelBinarizer().fit_transform(y_test)
    return X_train_dataset, X_test_dataset, y_train_lb, y_test_lb, labels

""" Neural Network classification """
# Using NumPy
def np_classification(X_train, X_test, y_train, y_test, labels, name):
    print("[INFO] training network...")
    input_shape = X_train.shape[1]
    nn = NeuralNetwork([input_shape, 64, 10])
    print(f"[INFO] {nn}")
    nn.fit(X_train, y_train, epochs = 10, displayUpdate = 1)
    predictions = nn.predict(X_test)
    yPred = predictions.argmax(axis=1)
    # make classification report
    report = classification_report(y_test.argmax(axis=1), yPred, target_names = labels)
    # save classification report
    model = "np"
    report_to_txt(report, name, model)
    # print classification report
    return print(report)

# Using TensorFlow
def tf_classification(X_train, X_test, y_train, y_test, labels, name):
    # create a sequential model
    model = Sequential()
    # define input shape
    in_shape = X_train.shape[1]
    # add layers one at a time
    model.add(Dense(256, input_shape = (in_shape,), activation = "relu"))
    model.add(Dense(128, activation = "relu"))
    model.add(Dense(10, activation = "softmax"))
    # define gradient descent
    sgd = SGD(0.01)
    # compile model
    model.compile(loss = "categorical_crossentropy",
                  optimizer = sgd,
                  metrics = ["accuracy"])
    # train model and save history
    history = model.fit(X_train, y_train, # what the model trains on
                    validation_data = (X_test, y_test), # unseen data the model tests on
                    epochs = 10,
                    batch_size = 32)
    # evaluate network
    predictions = model.predict(X_test, batch_size=32)
    # make classification report
    report = classification_report(y_test.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labels)
    # save classification report
    model = "tf"
    report_to_txt(report, name, model)
    # print classification report
    return print(report)

""" Main function """
def main():
    args = parse_args()
    # if the dataset is MNIST_784
    if args["dataset"] == "MNIST_784":
        # run loading and processing function for MNIST_784
        (X_train, X_test, y_train, y_test, labels) = MNIST_784()
        name = "MNIST_784"
        # if the model is tensorflow
        if args["model"] == "tensorflow":
            # run tensorflow neural network classification
            tf_classification(X_train, X_test, y_train, y_test, labels, name)
        # if the model is numpy
        elif args["model"] == "numpy":
            # run numpy neural network classification
            np_classification(X_train, X_test, y_train, y_test, labels, name)
        # otherwise, pass
        else:
            pass
    # if the dataset is CIFAR_10
    elif args["dataset"] == "CIFAR_10":
        # run loading and processing function for CIFAR_10
        (X_train, X_test, y_train, y_test, labels) = CIFAR_10()
        name = "CIFAR_10"
        # if the model is tensorflow
        if args["model"] == "tensorflow":
            # run tensorflow neural network classification
            tf_classification(X_train, X_test, y_train, y_test, labels, name)
        # if the model is numpy
        elif args["model"] == "numpy":
            # run numpy neural network classification
            np_classification(X_train, X_test, y_train, y_test, labels, name)
        # otherwise, pass
        else:
            pass
    # otherwise, pass 
    else:
        pass

if __name__=="__main__":
    main()