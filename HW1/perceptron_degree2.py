import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

data = pd.read_csv("s_dataset.csv", header=None)
di = {"M": 1,"R": 0}
data[60] = data[60].map(di)
y = data[60]
X = data.drop(60, axis=1)
X2 = X**2
data = pd.concat([X,X2,y], axis=1,ignore_index=True)
data.insert(0,-1,1)


def get_error(actual, predicted):
    return 1 - (actual == predicted).sum() / float(len(actual))

def get_confusion_matrix(actual, predicted):
    #find number of classes
    labels = np.unique(actual)
    #create labels x labels dataframe
    matrix = pd.DataFrame(np.zeros((len(labels), len(labels))))
    #(i,j) element of matrix is where actual lable is 'i' but predicted label is 'j'
    for i in range(len(labels)):
        for j in range(len(labels)):
            matrix.iloc[i, j] = np.sum((actual == labels[i]) & (predicted == labels[j]))
    return matrix

def train_test_valid_data(data,frac_train,frac_valid):
    m = data.shape[1]
    n_train = math.floor(frac_train*data.shape[0])
    n_valid = math.floor(frac_valid*data.shape[0])
    data = data.values
    train_data = data[:n_train,:]
    valid_data = data[n_train:n_train+n_valid,:]
    test_data = data[n_train+n_valid:,:]
    return train_data[:,:m-1],train_data[:,m-1],valid_data[:,:m-1],valid_data[:,m-1],test_data[:,:m-1],test_data[:,m-1]

X_train, y_train,X_valid, y_valid ,X_test, y_test = train_test_valid_data(data,0.7,0.2)

def activation_function(I):
    if I > 0:
        return 1
    if I <= 0:
        return 0

def check_single_sample(sample,weights,t):
    I = weights.T.dot(sample)
    y =activation_function(I)
    weights = weights + (t - y) * sample
    return weights

def check_all_samples(samples,weights,t_arr):
    for i in range(samples.shape[0]):
        weights = check_single_sample(samples[i,:],weights,t_arr[i])
    return weights

def prceptron(samples,t_arr,max_iter):
    iterate = 0
    weights = np.zeros(samples.shape[1])
    error_train = []
    mse_train = []
    error_valid = []
    mse_valid = []
    for i in range(max_iter):
        old_weights = weights.copy()
        weights = check_all_samples(samples,weights,t_arr)
        I_train = X_train.dot(weights)
        pred_train = np.array([activation_function(x) for x in I_train])
        error_train.append(get_error(y_train,pred_train))
        mse_train.append(np.sum((I_train-y_train)**2)/I_train.shape[0])
        I_valid = X_valid.dot(weights)
        pred_valid = np.array([activation_function(x) for x in I_valid])
        error_valid.append(get_error(y_valid, pred_valid))
        mse_valid.append(np.sum((I_valid - y_valid) ** 2) / I_valid.shape[0])
        if (np.sum(abs(weights - old_weights))) == 0.0:
            print(f"stopped after {i} iteration")
            break
    iterate = i
    plt.plot(range(iterate+1),error_train)
    plt.title("Train set - perceptron_degree2")
    plt.xlabel("Epoch")
    plt.ylabel("error")
    plt.show()
    plt.plot(range(iterate + 1), error_valid)
    plt.title("Validation set - perceptron_degree2")
    plt.xlabel("Epoch")
    plt.ylabel("error")
    plt.show()
    plt.plot(range(iterate + 1), mse_train)
    plt.title("Train set - perceptron_degree2")
    plt.xlabel("Epoch")
    plt.ylabel("mse")
    plt.show()
    plt.plot(range(iterate + 1), mse_valid)
    plt.title("Validation set - perceptron_degree2")
    plt.xlabel("Epoch")
    plt.ylabel("mse")
    plt.show()
    return weights,iterate

weights,iterate = prceptron(X_train,y_train,10000)

I_train = X_train.dot(weights)
pred_train = np.array([activation_function(x) for x in I_train])
accuracy_train = 1 - get_error(y_train,pred_train)
conf_train = get_confusion_matrix(y_train,pred_train)
print("accuracy for train set:",accuracy_train)
print("confusion matrix for train set:")
print(conf_train)

I_valid = X_valid.dot(weights)
pred_valid = np.array([activation_function(x) for x in I_valid])
accuracy_valid = 1 - get_error(y_valid, pred_valid)
conf_valid = get_confusion_matrix(y_valid,pred_valid)
print("accuracy for valid set:",accuracy_valid)
print("confusion matrix for valid set:")
print(conf_valid)
