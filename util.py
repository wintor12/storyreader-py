
def loadFeatures():
    return np.loadtxt('data/feature_train'),np.loadtxt('data/feature_val')

def loadUpvote():
    return np.loadtxt('data/y_train'),np.loadtxt('data/y_val')
