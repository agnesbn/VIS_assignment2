# Assignment 2 – Image classifier benchmark scripts
The portfolio for __Visual Analytics S22__ consists of 4 projects (3 class assignments and 1 self-assigned project). This is the __second assignment__ in the portfolio.

## 1. Contribution
The initial assignment was made partly in collaboration with others from the course, but the final code is my own. I made several adjustments to the code since I first handed it in during the course.

The `neuralnetwork.py` script in the `utils` folder was made by Ross.

## 2. Assignment description
When we were first assigned the assignment, the assignment description was as follows:

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

## 3. Methods


## 4. Usage
Before running the script, run the following in the Terminal:
```
pip install --upgrade pip
pip install scikit-learn tensorflow opencv-python
sudo apt-get update
sudo apt-get -y install graphviz
```
Then, from the `VIS_assignment2` directory, run:

### Logistic Regression model
```
python src/logistic_regression.py --dataset {DATASET}
```
Input:
- `{DATASET}` represents the given dataset you wish to train the model with. Here, you can put in either `CIFAR_10` or `MNIST_784`.


### Neural Network model
```
python src/nn_classifier.py --dataset {DATASET} --model {NN_MODEL}
```
Input:
- `{DATASET}` represents the given dataset you wish to train the model with. Here, you can put in either `CIFAR_10` or `MNIST_784`.
- `{NN_MODEL}` represents the given Neural Network model you wish to apply to the data. Here, you can put in either `numpy` or `tensorflow`.

## 5. Discussion of results

