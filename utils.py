import numpy as np


def preprocessFeatures(feature_train, feature_val):
    # add log view feature
    feature_train = \
        np.append(np.log(feature_train[:, 0]).reshape(-1, 1), feature_train, 1)
    feature_val = np.append(np.log(feature_val[:, 0]).reshape(-1, 1), feature_val, 1)
    # standardize features
    train_mean = np.mean(feature_train, 0)
    train_std = np.std(feature_train, 0)
    return (feature_train - train_mean) / train_std, \
        (feature_val - train_mean) / train_std


def loadFeatures(train=True):
    if train:
        return np.loadtxt('data/feature_train'), np.loadtxt('data/feature_val')
    else:
        return np.loadtxt('data/feature_train'), np.loadtxt('data/feature_test')


def loadUpvote(train=True):
    if train:
        return np.log(np.loadtxt('data/y_train') + 1), np.log(
            np.loadtxt('data/y_val') + 1)
    else:
        return np.log(np.loadtxt('data/y_train') + 1), np.log(
            np.loadtxt('data/y_test') + 1)
