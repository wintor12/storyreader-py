import codecs
import numpy as np


# load dictionary
voc = {}
with codecs.open('data/dictionary', 'r', 'utf-8') as dic:
    for line in dic:
        word, index = line.strip().split(' ')
        voc[index] = word


def processText(src, tgt):
    res = []
    with codecs.open('data/' + src, 'r', 'utf-8') as p:
        for line in p:
            res.append(' '.join([voc[x] for x in line.strip().split(' ')]))
    with codecs.open('data/' + tgt, 'w', 'utf-8') as p:
        p.write('\n'.join(res))


def processFeatures():
    feature_train, feature_val, feature_test = (np.loadtxt('data/feature_train'),
                                                np.loadtxt('data/feature_val'),
                                                np.loadtxt('data/feature_test'))

    # add log view feature
    feature_train = \
        np.append(np.log(feature_train[:, 0]).reshape(-1, 1), feature_train, 1)
    feature_val = np.append(np.log(feature_val[:, 0]).reshape(-1, 1), feature_val, 1)
    feature_test = np.append(np.log(feature_test[:, 0]).reshape(-1, 1), feature_test, 1)

    # standardize features
    train_mean = np.mean(feature_train, 0)
    train_std = np.std(feature_train, 0)
    feature_train = (feature_train - train_mean) / train_std
    feature_val = (feature_val - train_mean) / train_std
    feature_test = (feature_test - train_mean) / train_std

    np.savetxt('data/p_feature_train', feature_train)
    np.savetxt('data/p_feature_val', feature_val)
    np.savetxt('data/p_feature_test', feature_test)


def processUpvotes():
    y_train, y_val, y_test = (np.log(np.loadtxt('data/y_train') + 1),
                              np.log(np.loadtxt('data/y_val') + 1),
                              np.log(np.loadtxt('data/y_test') + 1))
    np.savetxt('data/p_y_train', y_train)
    np.savetxt('data/p_y_val', y_val)
    np.savetxt('data/p_y_test', y_test)


processText('question_train', 'q_train')
processText('question_val', 'q_val')
processText('question_test', 'q_test')
processText('story_train', 's_train')
processText('story_val', 's_val')
processText('story_test', 's_test')
processFeatures()
processUpvotes()
