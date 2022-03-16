import os
import pickle

import numpy as np
from tqdm import tqdm
from xkcdpass import xkcd_password as xp
from password_generator import PasswordGenerator
import pandas as pd


class XhcdPassGeneratorLocal(object):
    def __init__(self, min_length=5, max_length=8):
        wordfile = xp.locate_wordfile()
        self.mywords = xp.generate_wordlist(wordfile=wordfile, min_length=min_length, max_length=max_length)

    def generate(self, word):
        return xp.generate_xkcdpassword(self.mywords, acrostic=word)

    def generate_muil(self, words):
        passwords = []
        for word in words:
            passwords.append(self.generate(word))
        return passwords


class RandomPasGeneratorLocal(object):
    def __init__(self):
        self.pwo = PasswordGenerator()

    def generate(self):
        return self.pwo.generate()

    def generate_muil(self, num):
        passwords = []
        for i in tqdm(range(num)):
            passwords.append(self.generate())
        return np.array(passwords)


def process_found_pass(src):
    """
    remove some str that are not password from pass.csv
    :param src:
    :return:
    """
    data = pd.read_csv(f"{src}/pass.csv",
                       on_bad_lines='skip',
                       index_col=0)
    # which has " "(space)
    data = data[(data['str'].str.find(' ') == -1)]
    # which si smaller that 6
    data = data[data['str'].str.len() >= 6]
    # which var name is equaled with its str text
    data = data[~(data['var'].str.lower() == data['str'].str.lower())]

    # process
    data.reset_index(inplace=True)
    data = data.drop(columns='index')
    data = data.drop_duplicates()
    # save
    data.to_csv(f"{src}/pass.csv")


def generate_random_pass():
    """
    generate random password
    :return:
    """
    t = RandomPasGeneratorLocal()
    data: np.ndarray = t.generate_muil(100000)
    with open('../raw_dataset/random_pass.npy', 'wb') as f:
        np.save(f, data)


def remove_pass_from_string(src):
    """
    clear password in string.csv
    :param src:
    :return:
    """
    # string.csv
    str_text = pd.read_csv(f"{src}/string.csv", index_col=0)
    str_text.columns = ["var", 'str', 'line', 'location', 'project']
    # pass.csv
    pass_text = pd.read_csv(f"{src}/pass.csv", index_col=0)
    pass_text.columns = ["var", 'str', 'line', 'location', 'project']
    # remove the str has same value
    code_list = pass_text['location'].tolist()
    str_text = str_text[~str_text['location'].isin(code_list)]
    # Save
    str_text.to_csv(f"{src}/string.csv")


def merge_and_label():
    with open('../raw_dataset/random_pass.npy', 'rb') as f:
        randowm_pass = np.load(f, allow_pickle=True).tolist()
    with open('../raw_dataset/rockyou.npy', 'rb') as f:
        user_pass = np.load(f, allow_pickle=True).tolist()
        user_pass = [ap[0] for ap in user_pass]
    with open('../raw_dataset/nopass_str.npy', 'rb') as f:
        nopass = np.load(f, allow_pickle=True)

    data = []
    label = []
    for i, p in enumerate([randowm_pass, user_pass, nopass]):
        p = pd.DataFrame(p,dtype=str)
        p = p.dropna()
        p = p.to_numpy().reshape(-1).tolist()
        label.extend(np.zeros(len(p), dtype=int) + i)
        data.extend(p)
    data = pd.DataFrame(data, dtype=str)
    label = pd.DataFrame(label, dtype=int)
    with open('../dataset/data.pkl', 'wb') as f:
        pickle.dump(data, f)
    with open('../dataset/label.pkl', 'wb') as f:
        pickle.dump(label, f)



if __name__ == '__main__':
    # merge_and_label()
    # generate_no_pass_str('/home/rain/PassAna/csv')
    remove_pass_from_string('/home/rain/PassAna/csv/python')


