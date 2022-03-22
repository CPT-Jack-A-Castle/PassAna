import pickle

import pandas as pd

from context.contextClassifier import GAN


def load_pkl(src):
    with open(src,'rb') as f:
        data = pickle.load(f)
    return data

def generator(num, train=True):
    if train:
        X = load_pkl('./dataset/my_context_data.pkl').to_numpy().reshape(-1)
        Y = load_pkl('./dataset/my_context_label.pkl').to_numpy().reshape(-1)

        gan = GAN(padding_len=32)
        pass_data = X[Y==0]

        pass_data = gan.words2vec_generator(pass_data, fit=True)

        gan.create_model()

        gan.train(pass_data, epochs=3000, batch_size=32, sample_interval=100)
        gan.save_generator()
        return gan.generator_texts(num)
    else:
        gan = GAN(padding_len=32)
        gan.load_generator()
        return gan.generator_texts(num)

if __name__ == '__main__':
    gen_pass_context = generator(1000, False)
    gen_pass_context = pd.DataFrame(gen_pass_context,columns=['context'])
    gen_pass_context.to_csv('raw_dataset/mycontext_pass_gen.csv', index=False)