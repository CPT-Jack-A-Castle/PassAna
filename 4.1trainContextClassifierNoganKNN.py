import pickle
import numpy as np
from sklearn.model_selection import train_test_split

from context.contextClassifier import CNNClassifierGlove, KNNClassifier
from tokenizer.tool import load_pkl

if __name__ == '__main__':
    X = load_pkl('./dataset/nogan_train_data.pkl').reshape(-1)
    Y = load_pkl('./dataset/nogan_train_label.pkl').reshape(-1)

    knnClassifier = KNNClassifier(padding_len=256, glove_dim=100)

    X, Y = knnClassifier.words2vec(X, Y, fit=False)

    X, X_t, Y, Y_t = train_test_split(X, Y, stratify=Y, test_size=0.1)

    train_data, valid_data = [X, np.array(Y, dtype=int)], [X_t, np.array(Y_t, dtype=int)]

    knnClassifier.create_model()
    knnClassifier.sklearn_run(train_data)

    knnClassifier.save_model('model/context/model_nogan_knn.pkl')