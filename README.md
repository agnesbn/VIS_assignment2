# Assignment 2 – Image classifier benchmark scripts
The portfolio for __Visual Analytics S22__ consists of 4 projects (3 class assignments and 1 self-assigned project). This is the __second assignment__ in the portfolio.

## 1. Contribution
The initial assignment was made partly in collaboration with others from the course, but the final code is my own. I made several adjustments to the code since I first handed it in during the course.

The `neuralnetwork.py` script in the `utils` folder was made by Ross.
Help from https://medium.com/mlearning-ai/lenet-and-mnist-handwritten-digit-classification-354f5646c590.
Save history function is based on one made by Ross.

## 2. Assignment description
### Main task
For this assignment, you will take the classifier pipelines we covered in lecture 7 and turn them into *two separate ```.py``` scripts*. Your code should do the following:

- One script should be called ```logistic_regression.py``` and should do the following:
  - Load either the **MNIST_784** data or the **CIFAR_10** data
  - Train a Logistic Regression model using ```scikit-learn```
  - Print the classification report to the terminal **and** save the classification report to ```out/lr_report.txt```
- Another scripts should be called ```nn_classifier.py``` and should do the following:
  - Load either the **MNIST_784** data or the **CIFAR_10** data
  - Train a Neural Network model using the premade module in ```neuralnetwork.py```
  - Print output to the terminal during training showing epochs and loss
  - Print the classification report to the terminal **and** save the classification report to ```out/nn_report.txt```

### Bonus tasks
- Use ```argparse()``` so that the scripts use either **MNIST_784** or **CIFAR_10** based on some input from the user on the command line
- Use ```argparse()``` to allow users to define the number and size of the layers in the neural network classifier.
- Write the script in such a way that it can take either **MNIST_784** or **CIFAR_10** or **any data that the user wants to classify**
  - You can determine how the user data should be structured, by saying that it already has to be pre-processed or feature extracted.


## 3. Methods
### Main tasks
#### Logistic Regression model
The script, `logistic_regression.py`, uses different functions to load and process respectively the MNIST_784 and the CIFAR_10 datasets, so that they are returned in the same format, split, normalised and reshaped (if necessary). This data is then used as input in a logistic regression classifier function, which creates a model, predicts classes of the validation data based on the model, and creates and saves a classification report showing the main classification metrics: precision, recall, f1-score, and accuracy.

#### Neural Network model
The `nn_classifier.py`-script uses the same loading and processing function as the previously mentioned ones for MNIST_784 and CIFAR_10. The output from these functions are then used in either a NumPy or TensorFlow neural network classification model. The respective model is compiled and trained on the input data, the history is saved, and the final evaluation of the model is saved in the form of a classification report.

### Bonus tasks
As for the bonus tasks, I did not complete 2/3 but I did use ```argparse()``` to allow for the script to be used on either **MNIST_784** or **CIFAR_10** based on input from the command line. Though I did not allow for users to define the number and size of layers in the neural network classifier, I did allow for them to specify which neural network model to apply to the data, providing a number for different model types: **NumPy**, **TensorFlow**, **ShallowNet**, **LeNet**, and **VGG16**. With more time, the two final bonus tasks could be done by tweaking the scripts I provided here.

## 4. Usage
### Install packages
Before running the script, run the following in the Terminal:
```
pip install --upgrade pip
pip install *opencv-python *scikit-learn *tensorflow tensorboard tensorflow-hub pydot scikeras[tensorflow-cpu]
sudo apt-get update
sudo apt-get -y install graphviz
```

### Logistic Regression model
- Make sure to change the current directory to `VIS_assignment2` and then run:
```
python src/logistic_regression.py --dataset {DATASET}
```
- `{DATASET}` represents the given dataset you wish to train the model with. Here, you can put in either `CIFAR_10` or `MNIST_784`.
- The output is saved in `out/lr`.

### Neural Network model
- Make sure to change the current directory to `VIS_assignment2` and then run:
```
python src/nn_classifier.py --dataset {DATASET} --model {NN_MODEL}
```
- Input:
    - `{DATASET}` represents the given dataset you wish to train the model with. Here, you can put in either `CIFAR_10` or `MNIST_784`.
    - `{NN_MODEL}` represents the given Neural Network model you wish to apply to the data. Here, you can put in `numpy`, `tensorflow`, `shallownet`, `lenet`, or `vgg16`.
- The output is saved in `out/nn`.

## 5. Discussion of results
After running each model on both datasets for 15 epochs, the accuracy scores were:
| **Accuracy after<br>15 epochs** 	| **NumPy** 	| **TensorFlow** 	| **ShallowNet** 	| **LeNet** 	|
|---------------------------------	|-----------	|----------------	|----------------	|-----------	|
| **MNIST_784**                   	| 0.93      	| 0.91           	| 0.98           	| 0.97      	|
| **CIFAR_10**                    	| 0.37      	| 0.40           	| 0.56           	| 0.51      	|


