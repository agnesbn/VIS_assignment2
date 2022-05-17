"""
Image classification using a Neural Network model
"""
""" Import the relevant packages """
 # base tools
import sys,os
import datetime
import matplotlib.pyplot as plt
 # argument parser
import argparse
 # image processing
import cv2
 # neural networks with numpy
import numpy as np
sys.path.append(os.path.join("."))
from utils.neuralnetwork import NeuralNetwork
 # tools from tensorflow
import tensorflow as tf
from tensorflow.keras.utils import (to_categorical,
                                    plot_model)
from tensorflow.keras.datasets import (cifar10,
                                       mnist)
from tensorflow.keras.models import (Sequential,
                                     Model)
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense,
                                     BatchNormalization,
                                     Dropout)
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
 # machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_openml

""" Basic functions """
# Min-max normalisation function
def minmax(data):
    X_norm = (data-data.min())/(data.max()-data.min())
    return X_norm    

# Function to save classification report as TXT
def report_to_txt(report, name, model, short, epochs):
    outpath = os.path.join("out", "nn", f"nn_{name}_{short}_report.txt")
    with open(outpath,"w") as file:
        file.write(f"Neural Network - Classification report\nData: {name}\nModel: {model}\nEpochs: {epochs}\n")
        file.write(str(report))

# Argument parser
def parse_args():
    ap = argparse.ArgumentParser()
    # dataset argument (choose between mnist784 and cifar10)
    ap.add_argument("-d", 
                    "--dataset", 
                    required = True, 
                    help = "The dataset to train your model with; mnist784 or cifar10")
    # model argument (choose between numpy, tensorflow, shallownet, and lenet)
    ap.add_argument("-m",
                    "--model",
                    required = True,
                    help = "The neural network model you wish to train on your data; numpy, tensorflow, shallownet, or lenet")
    # number of epochs to train the model in
    ap.add_argument("-e",
                    "--epochs",
                    default=15,
                    type=int,
                    help = "The number of epochs to train your model in")
    args = vars(ap.parse_args())
    return args 

# Function to save history
def save_history(H, epochs, name, model, short):
    outpath = os.path.join("out", "nn", f"nn_{name}_{short}_history.png")
    plt.style.use("seaborn-colorblind")
    
    plt.figure(figsize=(12,6))
    plt.suptitle(f"History for {model} trained on {name}", fontsize=16)
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(outpath))


"""" Data loading and splitting functions """
# For the MNIST_784 dataset (used to train NumPy and TensorFlow models)
def MNIST_784_grey():
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
    name = "MNIST_784"
    return X_train_scaled, X_test_scaled, y_train_lb, y_test_lb, labels, name

# For the MNIST_784 dataset (used to train ShallowNet and LeNet models)
def MNIST_784_colour():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print('X_train shape', X_train.shape, 'X_test shape', X_test.shape)
    # get labels
    labels = sorted(set(y_train))
    # make labels into strings
    strings = []
    for l in labels:
        strings.append(str(l))
    labels = strings
    # reshape images
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    # convert image values from integers to floats
    X_train_float = X_train_reshaped.astype('float32')
    X_test_float = X_test_reshaped.astype('float32')
    # normalisation
    X_train_scaled = minmax(X_train_float)
    X_test_scaled = minmax(X_test_float)
    # binarise labels
    y_train_lb = LabelBinarizer().fit_transform(y_train)
    y_test_lb = LabelBinarizer().fit_transform(y_test)
    name = "MNIST_784"
    return X_train_scaled, X_test_scaled, y_train_lb, y_test_lb, labels, name
    
# For the CIFAR_10 dataset (used to train NumPy and TensorFlow models) 
def CIFAR_10_grey_reshaped():
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
    name = "CIFAR_10"
    return X_train_dataset, X_test_dataset, y_train_lb, y_test_lb, labels, name

# For the CIFAR_10 dataset (used to train ShallowNet and LeNet models)
def CIFAR_10_notgrey_notreshaped():
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
    # min-max normalisation
    X_train_scaled = minmax(X_train)
    X_test_scaled = minmax(X_test)
    # binarise labels
    y_train_lb = LabelBinarizer().fit_transform(y_train)
    y_test_lb = LabelBinarizer().fit_transform(y_test)
    name = "CIFAR_10"
    return X_train_scaled, X_test_scaled, y_train_lb, y_test_lb, labels, name


""" Neural Network classification """
# Using NumPy
def np_classification(X_train, X_test, y_train, y_test, labels, name, epochs):
    print("[INFO] training network...")
    input_shape = X_train.shape[1]
    nn = NeuralNetwork([input_shape, 64, 10])
    print(f"[INFO] {nn}")
    H = nn.fit(X_train, y_train, 
               epochs = epochs, 
               displayUpdate = 1)
    # evaluate network
    predictions = nn.predict(X_test)
    yPred = predictions.argmax(axis=1)
    # make classification report
    report = classification_report(y_test.argmax(axis=1), 
                                   yPred, 
                                   target_names = labels)
    # save classification report
    model = "NumPy"
    short = "np"
    report_to_txt(report, name, model, short, epochs)
    # print classification report
    return print(report)

# Using TensorFlow
def tf_classification(X_train, X_test, y_train, y_test, labels, name, epochs):
    # initialise the model
    model = Sequential()
    # define input shape
    input_shape = X_train.shape[1]
    # add layers one at a time
    model.add(Dense(256, input_shape = (input_shape,), activation = "relu"))
    model.add(Dense(128, activation = "relu"))
    model.add(Dense(10, activation = "softmax"))
    # define gradient descent
    sgd = SGD(0.01)
    # compile model
    model.compile(loss = "categorical_crossentropy",
                  optimizer = sgd,
                  metrics = ["accuracy"])
    # train model and save history
    H = model.fit(X_train, y_train, # what the model trains on
                  validation_data = (X_test, y_test), # unseen data that the model tests on
                  epochs = epochs,
                  batch_size = 32)
    # evaluate network
    predictions = model.predict(X_test, batch_size=32)
    # make classification report
    report = classification_report(y_test.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labels)
    # save classification report
    model = "TensorFlow"
    short = "tf"
    report_to_txt(report, name, model, short, epochs)
    # save history
    save_history(H, epochs, name, model, short)
    # print classification report
    return print(report)

# Using a ShalowNet architecture
def sn_classification(X_train, X_test, y_train, y_test, labels, name, epochs):
    # initialise the model
    model = Sequential()
    # define the convolutional and ReLU layer
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model.add(Conv2D(32, # size of the layer
                     (3,3), # kernel size
                     padding = "same", # zero padding (the default) – "same" because padding is applied all around the image
                     input_shape = input_shape))
    model.add(Activation("relu"))
    # fully-connected layer
    model.add(Flatten())
    # neural network architecture
    model.add(Dense(10)) # prediction layer (so, this model has no hidden layers)
    model.add(Activation("softmax"))
    # define the gradient descent
    sgd = SGD(learning_rate = 0.01)
    # compile model
    model.compile(loss = "categorical_crossentropy",
                  optimizer = sgd,
                  metrics = ["accuracy"])
    # train model
    H = model.fit(X_train, y_train, # what we train on
                  validation_data = (X_test, y_test), # the unseen data we test on
                  epochs = epochs,
                  batch_size = 32, # the model goes through the data in batches instead of in individual images
                 )
    # evaluate network
    predictions = model.predict(X_test, batch_size=32)
    # save classification report
    report = classification_report(y_test.argmax(axis=1),
                                   predictions.argmax(axis=1),
                                   target_names=labels)
    model = "ShallowNet"
    short = "sn"               
    report_to_txt(report, name, model, short, epochs)
    # save history
    save_history(H, epochs, name, model, short)
    # print classification report
    return print(report)

# Using a LeNet architecture
def ln_classification(X_train, X_test, y_train, y_test, labels, name, epochs):
    # initialise model
    model = Sequential()
    # first set of layers, CONV (32 3x3 kernels) => RELU => MAXPOOL
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model.add(Conv2D(32, (3,3), 
                     padding = "same", 
                     input_shape = input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2),
                           strides = (2,2)))
    # second set of layers, CONV (50 5x5 kernels) => RELU => MAXPOOL
    model.add(Conv2D(50, (5,5), 
                     padding = "same")) 
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2), 
                           strides = (2,2)))
    # FC => RELU
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))
    # softmax classifier
    model.add(Dense(len(labels)))
    model.add(Activation("softmax"))
    # define the gradient descent
    sgd = SGD(learning_rate = 0.001)
    # compile model
    model.compile(loss = "categorical_crossentropy",
                  optimizer = sgd,
                  metrics = ["accuracy"])
    # train model
    H = model.fit(X_train, y_train, # what we train on
                  validation_data = (X_test, y_test), # the unseen data that we test on
                  epochs = epochs,
                  batch_size = 32,
                  verbose = 1)
    # evaluate network
    predictions = model.predict(X_test, batch_size=32)
    # save classification report
    report = classification_report(y_test.argmax(axis=1),
                                   predictions.argmax(axis=1),
                                   target_names=labels)
    model = "LeNet"
    short = "ln"
    report_to_txt(report, name, model, short, epochs)
    # save history
    save_history(H, epochs, name, model, short)
    # print classification report
    return print(report)


""" Main function """
def main():
    # parse arguments
    args = parse_args()
    # get epochs argument
    epochs = args["epochs"]
    # if the dataset is MNIST_784
    if args["dataset"] == "mnist784":
        # if the model is tensorflow
        if args["model"] == "tensorflow":
            # run relevant loading and processing function for MNIST_784
            (X_train, X_test, y_train, y_test, labels, name) = MNIST_784_grey()
            # run tensorflow neural network classification
            tf_classification(X_train, X_test, y_train, y_test, labels, name, epochs)
        # if the model is numpy
        elif args["model"] == "numpy":
            # run relevant loading and processing function for MNIST_784
            (X_train, X_test, y_train, y_test, labels, name) = MNIST_784_grey()
            # run numpy neural network classification
            np_classification(X_train, X_test, y_train, y_test, labels, name, epochs)
        # if the model is shallownet
        elif args["model"] == "shallownet":
            # run relevant loading and processing function for MNIST_784
            (X_train, X_test, y_train, y_test, labels, name) = MNIST_784_colour()
            # run shallownet classification
            sn_classification(X_train, X_test, y_train, y_test, labels, name, epochs)
        elif args["model"] == "lenet":
            # run relevant loading and processing function for MNIST_784
            (X_train, X_test, y_train, y_test, labels, name) = MNIST_784_colour()
            # run lenet classification
            ln_classification(X_train, X_test, y_train, y_test, labels, name, epochs)
        # otherwise, pass
        else:
            pass
    # if the dataset is CIFAR_10
    elif args["dataset"] == "cifar10":
        # if the model is tensorflow
        if args["model"] == "tensorflow":
            # run loading and processing function for CIFAR_10 with greyscaling and reshaping
            (X_train, X_test, y_train, y_test, labels, name) = CIFAR_10_grey_reshaped()
            # run tensorflow classification
            tf_classification(X_train, X_test, y_train, y_test, labels, name, epochs)
        # if the model is numpy
        elif args["model"] == "numpy":
            # run loading and processing function for CIFAR_10 with greyscaling and reshaping
            (X_train, X_test, y_train, y_test, labels, name) = CIFAR_10_grey_reshaped()
            # run numpy neural network classification
            np_classification(X_train, X_test, y_train, y_test, labels, name, epochs)
        # if the model is shallownet
        elif args["model"] == "shallownet":
            # run loading and processing function for CIFAR_10 without greyscaling and reshaping
            (X_train, X_test, y_train, y_test, labels, name) = CIFAR_10_notgrey_notreshaped()
            # run shallownet classification
            sn_classification(X_train, X_test, y_train, y_test, labels, name, epochs)
        # if the model is lenet
        elif args["model"] == "lenet":
            # run loading and processing function for CIFAR_10 without greyscaling and reshaping
            (X_train, X_test, y_train, y_test, labels, name) = CIFAR_10_notgrey_notreshaped()
            # run lenet classification
            ln_classification(X_train, X_test, y_train, y_test, labels, name, epochs)
        # otherwise, pass
        else:
            pass
    # otherwise, pass 
    else:
        pass

if __name__=="__main__":
    main()