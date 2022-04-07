from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from context.contextClassifier import CNNClassifierGlove
from passwd.pwdClassifier import NgramPwdClassifier
import pandas as pd
import numpy as np


def first_model(X):
    X = X['str'].to_numpy().reshape(-1)
    ngramPwdClassifier = NgramPwdClassifier(padding_len=512, class_num=4)
    X, _ = ngramPwdClassifier.words2vec(X, n=3, fit=False)
    ngramPwdClassifier.load_model('model/pass/model_my_glove_4.h5')

    y_pred = ngramPwdClassifier.model.predict(X)
    return y_pred


def second_model(X):
    cnnContextClassifier = CNNClassifierGlove(padding_len=256)

    X, Y = cnnContextClassifier.words2vec(X, fit=False)
    cnnContextClassifier.load_model('model/context/model_my.h5')

    y_pred = cnnContextClassifier.model.predict(X)

    return y_pred


def draw_map(cf_matrix, label):
    ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,
                     fmt='.2%', cmap='Blues')

    ax.set_xlabel('Predicted Label',fontsize=16)
    ax.set_ylabel('True Label',fontsize=16)

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(label)
    ax.yaxis.set_ticklabels(label)


if __name__ == '__main__':
    credential = pd.read_csv('raw_dataset/mycontext_pass.csv').dropna().sample(300)
    credential['line'] = credential['line'].astype(int)
    ordinary = pd.read_csv('raw_dataset/mycontext_str.csv').sample(300)

    X = pd.concat([credential, ordinary])
    Y = np.r_[np.ones(credential.shape[0]), np.zeros(ordinary.shape[0])]

    first_mark = first_model(X).argmax(axis=1)
    first_mark = np.minimum(np.ones(first_mark.shape), first_mark)

    second_X = X[first_mark == 1]
    second_Y = Y[first_mark == 1]

    second_mark = second_model(second_X)

    second_mark = second_mark.argmax(axis=1)

    matrix = confusion_matrix(Y, second_mark)
    draw_map(matrix, ['Ordinary', 'Password'])
    plt.show()

    m = classification_report(Y, second_mark, digits=4)
    print(m)
