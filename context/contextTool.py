import os
import pickle

import pandas as pd
from tqdm import tqdm
import numpy as np

def read_csv_from_projects(src, header):
    """
    read csv file and rename their header
    :param src:  source of csv file path
    :param header: rename list
    :return:
    """
    csv_data = pd.read_csv(src, index_col=0)
    csv_data.columns=header
    return csv_data.drop_duplicates()


def split_context_csv_by_project(csv_str: pd.DataFrame, csv_context: pd.DataFrame):
    """
    merge the string csv with their context (if have)
    :param csv_str:  string data
    :param csv_context:  context data
    :return:
    """
    csv_data_by_group = csv_context.groupby(["var", "location"]).apply(_concat_context).reset_index()

    merge_csv = pd.merge(csv_str, csv_data_by_group, on=['var', 'location'], how='outer')

    return merge_csv


def _concat_context(data):
    """
    merge all context as one array split by the ";"
    :param data:
    :return:
    """
    return pd.Series({
        "context": ";".join(data["context"].unique())
    })


def merge_my_context(src, passorstr):
    dirs = os.listdir(src)

    merge_out = pd.DataFrame(columns=["var", "str","line","location","project","context"])
    # explore all dir
    for proj_dir in dirs:
        if not os.path.exists(f'{src}/{proj_dir}/mycontext_{passorstr}.csv'):
            continue
        data = pd.read_csv(f'{src}/{proj_dir}/mycontext_{passorstr}.csv', index_col=0)
        merge_out = pd.concat([merge_out, data])
    return merge_out


def merge_and_label(myofpassfinder):
    if myofpassfinder == "my":
        pass_context = pd.read_csv('raw_dataset/mycontext_pass.csv')
        str_context = pd.read_csv('raw_dataset/mycontext_str.csv')
    else:
        pass_context = pd.read_csv('raw_dataset/passfindercontext_pass.csv')
        str_context = pd.read_csv('raw_dataset/passfindercontext_str.csv')

    data = []
    label = []
    for i, p in enumerate([pass_context, str_context]):
        p = p.dropna()
        p = p.to_numpy().reshape(-1).tolist()
        label.extend(np.zeros(len(p), dtype=int) + i)
        data.extend(p)
    data = pd.DataFrame(data, dtype=str)
    label = pd.DataFrame(label, dtype=int)
    if myofpassfinder == "my":
        out_label = "my"
    else:
        out_label = "passfinder"

    with open(f'dataset/{out_label}_context_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    with open(f'dataset/{out_label}_context_label.pkl', 'wb') as f:
        pickle.dump(label, f)


if __name__ == '__main__':
    pass_csv = read_csv_from_projects("/home/rain/program/tmp/pass.csv",
                                      ["var", 'str', 'line', 'location', 'project'])
    csv_data = read_csv_from_projects("/home/rain/program/tmp/pass_context_to.csv",
                                      ['var', 'location', 'context', 'project'])
    split_context_csv_by_project(pass_csv, csv_data)
