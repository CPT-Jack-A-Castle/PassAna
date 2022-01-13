import pandas as pd
import numpy as np

from pwd.pwdPredictor import FastTextPredictor, HASHPredictor
from tokenizer.tool import train_valid_split


if __name__ == '__main__':
    # dataset = pd.read_csv('../dataset/rockyou.txt',
    #                       delimiter="\n",
    #                       header=None,
    #                       names=["Passwords"],
    #                       encoding="ISO-8859-1",
    #                       )
    #
    # # dataset = load_dataset('sst', 'default')
    #
    # # X = np.array(dataset.data['train'][0])
    # # Y = np.array(dataset.data['train'][1]).round()
    #
    # X = dataset.to_numpy().reshape(-1).tolist()
    # Y = np.ones(len(dataset))
    #
    # fastTextPredictor = FastTextPredictor(padding_len=128, class_num=2)
    # X, Y = fastTextPredictor.words2vec(X, Y, False)
    # train_data, valid_data = train_valid_split(X, Y)
    #
    # fastTextPredictor.create_model()
    # fastTextPredictor.run(train_data, valid_data, epochs=25, batch_size=64)

    dataset = pd.read_csv('./dataset/rockyou.txt',
                          delimiter="\n",
                          header=None,
                          names=["Passwords"],
                          encoding="ISO-8859-1",
                          nrows =10000
                          )
    X = dataset.to_numpy().reshape(-1).tolist()
    Y = np.ones(len(dataset))

    md5Predictor = HASHPredictor(padding_len=128, class_num=2)

    X, Y = md5Predictor.words2vec(X, Y)


    md5Predictor.create_model()
    X = HASHPredictor.data2NNform(X)

    md5Predictor.run([X,Y], [X,Y], epochs=25, batch_size=64)