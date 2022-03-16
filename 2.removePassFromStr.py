import os

import pandas as pd

from passwd.passTool import remove_pass_from_string

if __name__ == '__main__':
    src = "/home/rain/PassAna/csv"
    dirs = os.listdir(src)
    # list all dir
    for language_dir in dirs:
        remove_pass_from_string(f"{src}/{language_dir}")

