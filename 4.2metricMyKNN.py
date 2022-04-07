import pickle

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from context.contextClassifier import CNNClassifierGlove, KNNClassifier
from passwd.pwdClassifier import NgramPwdClassifier
import numpy as np

from tokenizer.tool import load_pkl


def draw_map(cf_matrix, label):
    ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,
                     fmt='.2%', cmap='Blues')

    ax.set_xlabel('Predicted Label',fontsize=16)
    ax.set_ylabel('True Label',fontsize=16)

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(label)
    ax.yaxis.set_ticklabels(label)

    ## Display the visualization of the Confusion Matrix.
    # plt.show()

if __name__ == '__main__':
    X = load_pkl('./dataset/nogan_test_data.pkl').reshape(-1)
    Y = load_pkl('./dataset/nogan_test_label.pkl').reshape(-1)

    knnClassifier = KNNClassifier(padding_len=256)
    knnClassifier.create_model()

    X, Y = knnClassifier.words2vec(X, Y, fit=False)
    knnClassifier.load_model('model/context/model_my_knn.pkl')

    y_pred = knnClassifier.model.predict(X)

    Y = Y.argmax(axis=1)
    y_pred = y_pred.argmax(axis=1)

    matrix = confusion_matrix(Y, y_pred)
    draw_map(matrix, ['Ordinary', 'Password'])
    plt.show()
    # print()

    m = classification_report(Y, y_pred, digits=4)
    print(m)
