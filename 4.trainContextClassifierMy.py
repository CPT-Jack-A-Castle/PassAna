import pickle
import numpy as np
from sklearn.model_selection import train_test_split

from context.contextClassifier import CNNClassifierGlove


def load_pkl(src):
    with open(src,'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == '__main__':
    X = load_pkl('./dataset/my_context_data.pkl').to_numpy().reshape(-1)
    Y = load_pkl('./dataset/my_context_label.pkl').to_numpy().reshape(-1)

    cnnContextClassifier = CNNClassifierGlove(padding_len=128)

    X, Y = cnnContextClassifier.words2vec(X, Y, fit=True)
    X, X_t, Y, Y_t = train_test_split(X, Y, stratify=Y, test_size=0.1)

    cnnContextClassifier.get_matrix_6b(f"/home/rain/glove")

    train_data, valid_data = [X, np.array(Y, dtype=int)], [X_t, np.array(Y_t, dtype=int)]

    #train_data, valid_data = train_valid_split(X, Y)

    cnnContextClassifier.create_model()
    cnnContextClassifier.run(train_data, valid_data, epochs=100, batch_size=128)

    cnnContextClassifier.save()