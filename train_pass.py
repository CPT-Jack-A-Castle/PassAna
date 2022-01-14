import pickle

import pandas as pd
import numpy as np
from keras.utils import to_categorical

from pwd.pwdClassifier import FastTextPwdClassifier, HASHPwdClassifier, NgramPwdClassifier
from tokenizer.tool import train_valid_split


def pwdHash():
    dataset = pd.read_csv('./dataset/rockyou.txt',
                          delimiter="\n",
                          header=None,
                          names=["Passwords"],
                          encoding="ISO-8859-1",
                          # nrows =1000
                          )
    X = dataset.to_numpy().reshape(-1).tolist()

    # X = np.load('./dataset/pwd.npy')
    Y = np.ones(len(X))
    Y = to_categorical(Y)

    md5Predictor = HASHPwdClassifier(padding_len=128, class_num=2)

    X, Y = md5Predictor.words2vec(X, Y)

    np.save('./dataset/pwd.npy', X)

def randomHash():
    dataset = pd.read_csv('./dataset/random.csv',
                header=0,
                # encoding="ISO-8859-1",
                nrows =15000000
                )['sourcestring']

    md5Predictor = HASHPwdClassifier(padding_len=128, class_num=2)

    X = dataset.to_numpy().reshape(-1)
    Y = np.zeros(len(X))
    Y = to_categorical(Y)

    X, Y = md5Predictor.words2vec(X, Y)

    np.save('./dataset/randstr.npy', X)


def pwdNgram():

    X = np.load('./dataset/pwd&str.npy')
    Y = np.load('./dataset/pwd&label.npy')

    ngramPwdClassifier = NgramPwdClassifier(padding_len=128, class_num=2)
    X, Y = ngramPwdClassifier.words2vec(X, Y, fit=False)

    ngramPwdClassifier.get_matrix_6b(f"D:\\program\\glove.6B")
    train_data, valid_data = train_valid_split(X, Y)

    ngramPwdClassifier.create_model()
    ngramPwdClassifier.run(train_data, valid_data, epochs=25, batch_size=64)


def pwdNlp():
    dataset = pd.read_csv('./dataset/rockyou.txt',
                          delimiter="\n",
                          header=None,
                          names=["Passwords"],
                          encoding="ISO-8859-1",
                          nrows=1000
                          )

    X = dataset.to_numpy().reshape(-1).tolist()
    Y = np.ones(len(dataset))

    fastTextPredictor = FastTextPwdClassifier(padding_len=128, class_num=2)
    X, Y = fastTextPredictor.words2vec(X, Y, False)
    train_data, valid_data = train_valid_split(X, Y)

    fastTextPredictor.create_model()
    fastTextPredictor.run(train_data, valid_data, epochs=25, batch_size=64)

def classifier():
    md5Predictor = HASHPwdClassifier(padding_len=128, class_num=2)

    positive_data = np.load('./dataset/pwd.npy')
    positive_label = np.ones(positive_data.shape[0])

    negative_data = np.load('./dataset/randstr.npy')
    negative_label = np.zeros(negative_data.shape[0])

    X = np.r_[positive_data, negative_data]
    # X = HASHPwdClassifier.data2NNform(data)
    label = np.r_[positive_label, negative_label]
    Y = to_categorical(label)
    train_data, valid_data = train_valid_split(X, Y)


    md5Predictor.create_model()

    md5Predictor.run(train_data, valid_data, epochs=100, batch_size=128)



if __name__ == '__main__':
    pwdNgram()




