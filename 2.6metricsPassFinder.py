import pickle
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from passwd.passFinderPass import PassFinderPassClassifier
from passwd.pwdClassifier import NgramPwdClassifier
import numpy as np

def load_pkl(src):
    with open(src, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pkl(src, obj):
    with open(src, 'wb') as f:
        pickle.dump(obj, f)


def draw_map(cf_matrix, label):
    ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,
                     fmt='.2%', cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(label)
    ax.yaxis.set_ticklabels(label)

    ## Display the visualization of the Confusion Matrix.
    #plt.show()

if __name__ == '__main__':
    X = load_pkl('./dataset/pwd_test_data.pkl').reshape(-1)
    Y = load_pkl('./dataset/pwd_test_label.pkl').reshape(-1)

    passFinderPassClassifier = PassFinderPassClassifier(padding_len=128, class_num=4)
    X, Y = passFinderPassClassifier.words2vec(X, Y, fit=False)
    passFinderPassClassifier.load_model('model/pass/model_passfinder_4.h5')

    y_pred = passFinderPassClassifier.model.predict(X)

    Y = Y.argmax(axis=1)
    y_pred = y_pred.argmax(axis=1)

    m = classification_report(Y, y_pred, digits=4)
    print(m)

    matrix = confusion_matrix(Y, y_pred)
    draw_map(matrix, ['Ordinary', 'Random', 'Human', 'Tokens'])
    # plt.savefig('metrics/passfinder_confusion_matrix_4.pdf')
    # plt.clf()

    b_Y = np.minimum(np.ones(Y.shape), Y)
    b_y_pred = np.minimum(np.ones(Y.shape), y_pred)

    m = classification_report(b_Y, b_y_pred, digits=6)
    print(m)
