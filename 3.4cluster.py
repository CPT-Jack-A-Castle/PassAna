import pickle

import numpy as np
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

from context.contextClassifier import CNNClassifierGlove, camel_case_split
from tokenizer.tool import MyTokenizer


def load_pkl(src):
    with open(src, 'rb') as f:
        data = pickle.load(f)
    return data


def words2vec(texts, labels, fit=True):
    """
    Map raw texts to int vector
    :param texts: [ [], [],..., [] ]
    :param labels:[]
    :param fit: Whether refit the tokenizer
    :return: texts, labels
    """
    tokenizer = MyTokenizer()
    for i in range(len(texts)):
        tmp_text = texts[i].replace(".", " ").replace(";", " ").replace("_", " ")
        tmp_text = camel_case_split(tmp_text)
        texts[i] = " ".join(tmp_text)

    if fit:
        tokenizer.fit_on_texts(texts)
        tokenizer.save_tokenizer("tokenizer/context.pkl")
    else:
        tokenizer.load_tokenizer("tokenizer/context.pkl")

    # integer encode the documents
    encoded_docs = tokenizer.texts_to_sequences(texts)
    # pad documents to a max length of 128 words
    texts = pad_sequences(encoded_docs, maxlen=512)

    return texts, labels


if __name__ == '__main__':
    pass_context = pd.read_csv('raw_dataset/mycontext_pass.csv')["context"].sample(300)
    gen_context = pd.read_csv('raw_dataset/mycontext_pass_gen.csv')["context"].sample(300)
    str_context = pd.read_csv('raw_dataset/mycontext_str.csv')["context"].sample(8000)

    data = []
    label = []
    for i, p in enumerate([str_context, pass_context, gen_context]):
        p = p.dropna()
        p = p.to_numpy().reshape(-1).tolist()
        label.extend(np.zeros(len(p), dtype=int) + i)
        data.extend(p)
    X = pd.DataFrame(data, dtype=str).to_numpy().reshape(-1)
    Y = pd.DataFrame(label, dtype=int).to_numpy().reshape(-1)

    X, Y = words2vec(X, Y, fit=False)

    lda = PCA(n_components=2)
    lda.fit(X, Y)
    X_new = lda.transform(X)

    for i, name in enumerate(["Ordinary Text Context", "Credential Context", "GAN Generator Context"]):
        index = (Y == i)
        plt.scatter(X_new[index, 0], X_new[index, 1],
                    marker='.', label=name, s=5)

    plt.legend(prop={'size': 12})
    plt.show()
