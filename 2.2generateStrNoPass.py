import os
import pandas as pd

from passwd.passTool import three_sigma_deduce

if __name__ == '__main__':
    """
    合并生成常规的str，即不包含password 的纯粹的str, 且进行简单的数据处理
    """
    src = "/home/rain/PassAna/csv"
    dirs = os.listdir(src)
    init_data = pd.DataFrame(columns=['str'])
    for language_dir in dirs:
        dir_data = pd.read_csv(f"{src}/{language_dir}/string.csv", index_col=0)[['str']]
        init_data = pd.concat([init_data, dir_data], ignore_index=True)
    df2 = pd.read_csv('raw_dataset/password.csv')
    df2.columns = ['str']

    init_data = init_data.drop_duplicates(ignore_index=True)
    # length > 6
    init_data = init_data[init_data['str'].str.len() >= 6]
    # length < 32
    # init_data = init_data[init_data['str'].str.len() <= 40]

    intersected_df = pd.merge(init_data, df2, how='inner')

    init_data = init_data[~init_data['str'].isin(intersected_df['str'].tolist())]

    init_data.to_csv("raw_dataset/nopass_str.csv", index=False)

    three_sigma_deduce("raw_dataset/nopass_str.csv")