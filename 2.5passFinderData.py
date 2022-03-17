import pickle
import numpy as np
from sklearn.model_selection import train_test_split

from passwd.passFinderPass import PassFinderPassClassifier


def load_csv(src):
    with open(src,'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == '__main__':
    # x_train = np.random.randint(100,size=(1200,100))
    # y_train = np.random.randint(100,size=(1200,1))

    X = load_csv('./dataset/data.pkl').to_numpy().reshape(-1)
    Y = load_csv('./dataset/label.pkl').to_numpy().reshape(-1)

    passFinderClassifier = PassFinderPassClassifier(padding_len=128, class_num=4)

    X, Y = passFinderClassifier.words2vec(X, Y, fit=False)

    X, X_t, Y, Y_t = train_test_split(X, Y, stratify=Y, test_size=0.1)

    train_data, valid_data = [X, np.array(Y, dtype=int)], [X_t, np.array(Y_t, dtype=int)]

    passFinderClassifier.create_model()
    passFinderClassifier.run(train_data, valid_data, epochs=100, batch_size=128)