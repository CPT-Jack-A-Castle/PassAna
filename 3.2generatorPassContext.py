import pickle
import nltk
import numpy as np
import pandas as pd

from context.contextClassifier import GAN


def load_pkl(src):
    with open(src,'rb') as f:
        data = pickle.load(f)
    return data


def generator(num, train=True):
    padding_len = 32
    if train:
        pass_context = pd.read_csv('raw_dataset/mycontext_pass.csv')["context"].dropna().to_numpy()
        # str_context = pd.read_csv('raw_dataset/mycontext_str.csv')["context"].dropna()
        # str_context = str_context[str_context.str.len() >= 8].to_numpy()
        # merge_context = np.r_[pass_context, str_context]

        gan = GAN(padding_len=padding_len)

        gan.words2vec_tokenizer(pass_context, fit=True)

        # to vector
        pass_data = gan.words2vec_text(pass_context)
        # str_context = gan.words2vec_text(str_context)

        # labels = np.r_[np.ones(pass_context.shape), np.zeros(str_context.shape)]

        gan.create_model()

        gan.train(pass_data, epochs=2000, batch_size=128, sample_interval=100)
        gan.save_generator()
        return gan.generator_texts(num)
    else:
        gan = GAN(padding_len=padding_len)
        gan.load_generator()
        return gan.generator_texts(num)


if __name__ == '__main__':
    gen_pass_context = generator(30000, False)
    gen_pass_context = pd.DataFrame(gen_pass_context, columns=['context'])
    gen_pass_context = gen_pass_context.dropna()
    gen_pass_context.to_csv('raw_dataset/mycontext_pass_gen.csv', index=False)

    gen_pass_context = pd.read_csv('raw_dataset/mycontext_pass_gen.csv').dropna()
    gen_pass_context.to_csv('raw_dataset/mycontext_pass_gen.csv', index=False)