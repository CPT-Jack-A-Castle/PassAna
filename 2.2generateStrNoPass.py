import os
import pandas as pd


if __name__ == '__main__':
    src = "/home/rain/PassAna/csv"
    dirs = os.listdir(src)
    init_data = pd.DataFrame(columns=['str'])
    for language_dir in dirs:
        dir_data = pd.read_csv(f"{src}/{language_dir}/string.csv", index_col=0)[['str']]
        init_data = pd.concat([init_data, dir_data], ignore_index=True)
    init_data = init_data.drop_duplicates(ignore_index=True)
    init_data.to_csv("raw_dataset/nopass_str.csv", index=False)