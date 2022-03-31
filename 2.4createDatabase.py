# coding: utf-8
import pickle

from sklearn.model_selection import train_test_split

from passwd.passTool import merge_and_label

def load_pkl(src):
    with open(src, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pkl(src, obj):
    with open(src, 'wb') as f:
        pickle.dump(obj, f)


def split_train_and_test():
    X = load_pkl('./dataset/pwd_data.pkl').to_numpy().reshape(-1)
    Y = load_pkl('./dataset/pwd_label.pkl').to_numpy().reshape(-1)

    X, X_t, Y, Y_t = train_test_split(X, Y, stratify=Y, test_size=0.1)

    save_pkl('./dataset/pwd_train_data.pkl', X)
    save_pkl('./dataset/pwd_train_label.pkl', Y)

    save_pkl('./dataset/pwd_test_data.pkl', X_t)
    save_pkl('./dataset/pwd_test_label.pkl', Y_t)


if __name__ == '__main__':
    merge_and_label()
    split_train_and_test()