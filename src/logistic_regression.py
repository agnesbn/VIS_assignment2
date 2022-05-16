"""
Image classification using a Logistic Regression model
"""
# Import the relevant packages
 # base tools
import os, sys
import argparse
 # data analysis
import pandas as pd
import numpy as np
 # sklearn metrics
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
 # tensorflow
from tensorflow.keras.datasets import cifar10
 # image processing
import cv2

""" Basic functions """

# Min-max normalisation function
def minmax(data):
    X_norm = (data-data.min())/(data.max()-data.min())
    return X_norm    

# Function to save classification report as TXT
def report_to_txt(report, name):
    outpath = os.path.join("out", "lr", f"report_{name}.txt")
    with open(outpath,"w") as file:
        file.write(str(report))

# Argument parser
def parse_args():
    ap = argparse.ArgumentParser()
    # dataset argument (choose between MNIST_784 and CIFAR_10)
    ap.add_argument("-d", 
                    "--dataset", 
                    required = True, 
                    help = "The dataset to train your model with, MNIST_784 or CIFAR_10")
    args = vars(ap.parse_args())
    return args 

"""" Data loading and processing functions """
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
    return X_train_scaled, X_test_scaled, y_train, y_test, labels
 
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
    return X_train_dataset, X_test_dataset, y_train, y_test, labels

""" Logistic Regression classifier """
def train_log_model(X_train, X_test, y_train, y_test, labels, name):
    # create logistic regression model
    clf = LogisticRegression(penalty = "none",
                             tol = 0.1,
                             solver = "saga",
                             multi_class = "multinomial").fit(X_train, y_train)
    # get predictions
    y_pred = clf.predict(X_test)
    # make classification report
    report = metrics.classification_report(y_test, y_pred)
    report_to_txt(report, name)
    return print(report)

        
""" Main function """
def main():
    args = parse_args()
    # if the dataset is MNIST_784
    if args["dataset"] == "MNIST_784":
        (X_train, X_test, y_train, y_test, labels) = MNIST_784()
        name = "MNIST_784"
        report = train_log_model(X_train, X_test, y_train, y_test, labels, name)
    # else, if the dataset is CIFAR_10
    elif args["dataset"] == "CIFAR_10":
        (X_train, X_test, y_train, y_test, labels) = CIFAR_10()
        name = "CIFAR_10"
        report = train_log_model(X_train, X_test, y_train, y_test, labels, name)
    
    else:
        pass

if __name__=="__main__":
    main()