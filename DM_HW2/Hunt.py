import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def importdata():
    train_data = pd.read_csv(
        'venv/noisy_train.csv',
        sep=',', header=None)
    test_data = pd.read_csv(
        'venv/noisy_test.csv',
        sep=',', header=None)
    valid_data = pd.read_csv(
        'venv/noisy_valid.csv',
        sep=',', header=None)
    # Printing the dataswet shape 
#    print("Dataset Lenght: ", len(train_data))
#    print("Dataset Shape: ", train_data.shape)


    #print("Dataset: ", train_data.head())
    return train_data,test_data,valid_data


def splitdataset(data_train,data_test):
    # Seperating the target variable
    X_train = data_train.values[1:, 1:]
    y_train = data_train.values[1:, 0]
    #print("y_train: ", y_train[:])

    X_test = data_test.values[1:, 1:]
    y_test = data_test.values[1:, 0]

    return X_train, X_test, y_train, y_test


def train_using_gini(X_train, X_test, y_train):
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini",
                                      random_state=100, max_depth=3, min_samples_leaf=5)

    clf_gini.fit(X_train, y_train)
    return clf_gini


def tarin_using_entropy(X_train, X_test, y_train):
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=100,
        max_depth=3, min_samples_leaf=5)

    # Performing training 
    clf_entropy.fit(X_train, y_train)
    return clf_entropy


def prediction(X_test, clf_object):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))

    print("Accuracy : ",
          accuracy_score(y_test, y_pred) * 100)

    print("Report : ",
          classification_report(y_test, y_pred))


# Driver code
def main():
    # Building Phase
    data_train,data_test,data_valid = importdata()
    X_train, X_test, y_train, y_test = splitdataset(data_train,data_test)
    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train)

    # Operational Phase 
    print("Results Using Gini Index:")

    # Prediction using gini 
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)

    print("Results Using Entropy:")
    # Prediction using entropy 
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)


if __name__ == "__main__":
    main() 