import pickle

import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from passwd.pwdClassifier import HASHPwdClassifier, NgramPwdClassifier, FastTextPwdClassifier
from tokenizer.tool import train_valid_split




def load_csv(src):
    with open(src,'rb') as f:
        data = pickle.load(f)
    return data


def pwdNgram():
    X = load_csv('./dataset/pwd_data.pkl').to_numpy().reshape(-1)
    Y = load_csv('./dataset/pwd_label.pkl').to_numpy().reshape(-1)

    ngramPwdClassifier = NgramPwdClassifier(padding_len=128, class_num=4)

    X, Y = ngramPwdClassifier.words2vec(X, Y, fit=True)
    X, X_t, Y, Y_t = train_test_split(X, Y, stratify=Y, test_size=0.1)

    ngramPwdClassifier.get_matrix_6b(f"/home/rain/glove")

    train_data, valid_data = [X, np.array(Y, dtype=int)], [X_t, np.array(Y_t, dtype=int)]

    #train_data, valid_data = train_valid_split(X, Y)

    ngramPwdClassifier.create_model()
    ngramPwdClassifier.run(train_data, valid_data, epochs=100, batch_size=128)

    ngramPwdClassifier.save()

def pwdNlp():
    dataset = pd.read_csv('raw_dataset/rockyou.txt',
                          delimiter="\n",
                          header=None,
                          names=["Passwords"],
                          encoding="ISO-8859-1",
                          nrows=1000
                          )

    X = dataset.to_numpy().reshape(-1).tolist()
    Y = np.ones(len(dataset))

    fastTextPredictor = FastTextPwdClassifier(padding_len=128, class_num=3)
    X, Y = fastTextPredictor.words2vec(X, Y, False)
    train_data, valid_data = train_valid_split(X, Y)

    fastTextPredictor.create_model()
    fastTextPredictor.run(train_data, valid_data, epochs=25, batch_size=64)


def classifier():
    md5Predictor = HASHPwdClassifier(padding_len=128, class_num=3)

    X = load_csv('./dataset/data.pkl').to_numpy().reshape(-1)
    Y = load_csv('./dataset/label.pkl').to_numpy().reshape(-1)

    Y = to_categorical(Y)
    train_data, valid_data = train_valid_split(X, Y)
    md5Predictor.create_model()

    md5Predictor.run(train_data, valid_data, epochs=100, batch_size=128)



if __name__ == '__main__':
    pwdNgram()




