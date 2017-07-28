import numpy as np

def loadFeatures():
    return np.loadtxt('data/feature_train'),np.loadtxt('data/feature_val')

def loadUpvote():
    return np.log(np.loadtxt('data/y_train')+1),np.log(np.loadtxt('data/y_val')+1)
