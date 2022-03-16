import pandas as pd
from tqdm import tqdm


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


if __name__ == '__main__':
    pass_csv = read_csv_from_projects("/home/rain/program/tmp/pass.csv",
                                      ["var", 'str', 'line', 'location', 'project'])
    csv_data = read_csv_from_projects("/home/rain/program/tmp/pass_context_to.csv",
                                      ['var', 'location', 'context', 'project'])
    split_context_csv_by_project(pass_csv, csv_data)
