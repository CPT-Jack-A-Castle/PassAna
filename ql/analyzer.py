# coding: utf-8
import logging
import json
import string
import re
import pandas
import pexpect
import os
import csv
import pandas as pd

import http.client
import json

from tqdm import tqdm
from urllib3 import response


class Analyzer(object):

    def __init__(self, debug=False):
        # Analyze task handle
        self._analyze_task = None
        # Ql select task handle
        self._ql_task = None
        # Language Type
        self.language_type = None
        # default cmd
        self.cmd = 'cmd'

        if debug:
            logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                level=logging.INFO)

    def set_cmd(self, cmd):
        if cmd not in ['context_from', 'context_to', 'findString', 'checkRemote']:
            logging.error(f'Not support {cmd}!')
            return
        self.cmd = cmd

    def database_create(self, src, external_cmd=None):
        """
        Create database for a project
        :param src: file path
        :param external_cmd: additional command. For instance, Cpp may use 'make' to build the project.
        :return: True or False
        """

        # If a file has been analyzed
        if os.path.exists(f"{src}/{self.language_type}_database"):
            if os.path.exists(f"{src}/{self.language_type}_database/src.zip"):
                logging.info(f'[{src}] has been analyzed.')
                return False

        # analyze command
        cmd = f"codeql database create " \
              f"{self.language_type}_database --language={self.language_type}"
        if external_cmd is not None:
            cmd.join(f" --command{external_cmd}")
        self._analyze_task = pexpect.spawn(cmd, cwd=src)

        # Start analyze
        while True:
            line = self._analyze_task.readline().decode()
            logging.debug(line)
            if line.startswith('Successfully created database'):
                logging.info(f'Successfully created database')
                break
            if line.startswith('A fatal error'):
                logging.error(f'A fatal error')
                break
        return True

    def ql_find_str(self, src, skip=False, threads=4):
        """
        Run Ql file to analyze the database
        :param skip:
        :param src: file path
        :param threads: thread for running (default=1)
        """

        if skip and os.path.exists(f"{src}/results/getting-started/codeql-extra-queries-{self.language_type}/{self.cmd}.bqrs"):
            logging.info(f' It has been analyzed.')
            return False

        # analyze ql command
        cmd = f"codeql database analyze  " \
              f"{src} ql/{self.language_type}/findString.ql " \
              f"--format=csv --output={src}/result.csv --rerun --threads {threads}"

        self._ql_task = pexpect.spawn(cmd, timeout=300)
        while True:
            line = self._ql_task.readline().decode()
            logging.debug(line)
            if line.startswith('Interpreting results'):
                logging.info(f'Interpreting results')
                return True
            if line.startswith('A fatal error'):
                logging.error(f'A fatal error')
                return False
            if line.startswith('Error: Can only run queries'):
                logging.error(f'Error: Can only run queries')
                return False

    def ql_str_context(self, src, skip=False, threads=4):
        """
        run ql analyze to find string context flow
        :param src:
        :param skip:
        :param threads:
        :return:
        """
        if skip and os.path.exists(f"{src}/results/getting-started/codeql-extra-queries-{self.language_type}/{self.cmd}.bqrs"):
            logging.info(f'{src} has been analyzed.')
            return False

        # analyze ql command
        cmd = f"codeql database analyze  " \
              f"{src} ql/{self.language_type}/{self.cmd}.ql " \
              f"--format=csv --output={src}/{self.cmd}.csv --rerun --threads {threads}"

        # handler running
        self._ql_task = pexpect.spawn(cmd, timeout=3600)
        while True:
            line = self._ql_task.readline().decode()
            logging.debug(line)
            if line.startswith('Interpreting results'):
                logging.info(f'Interpreting results')
                return True
            if line.startswith('A fatal error'):
                logging.error(f'A fatal error')
                return False

    def specific_str_context_array(self, proj_path: str, source_path: str):
        csv_data = pandas.read_csv(source_path, index_col=0)
        # group by project
        csv_data_by_group = csv_data.groupby('project')

        for group, group_item in tqdm(csv_data_by_group):
            self.specific_str_context(proj_path, group, group_item['var'].tolist())

    def specific_str_context(self, proj_path: str, group: str, group_item: list):
        complete_path = proj_path + '/' + group
        # read
        with open(f'ql/{self.language_type}/{self.cmd}.ql', 'r') as f:
            lines = f.readlines()
            ql_code = lines[16]

        str_array = re.findall('\[.*\]',ql_code)[0]
        new_line = ql_code.replace(str_array, str(group_item).replace("'", '"'))
        lines[16] = new_line

        # replace
        with open(f'ql/{self.language_type}/{self.cmd}.ql', 'w') as f:
            f.writelines(lines)

        self.ql_str_context(complete_path)

    def decode_bqrs2json(self, src):
        """
        Decode the .bqrs file to json
        :param src:
        :return: result
        """
        path = f"{src}/results/getting-started/codeql-extra-queries-{self.language_type}/{self.cmd}.bqrs"
        # analyze ql command
        cmd = f"codeql bqrs decode --format=json --output={src}/{self.cmd}.json {path}"
        os.system(cmd)

        with open(f'{src}/out.json', 'r') as f:
            out = json.load(f)
        return out

    def decode_bqrs2csv(self, src):
        """
        Decode the .bqrs file to csv
        :param src:
        :return: result
        """
        path = f"{src}/results/" \
               f"getting-started/codeql-extra-queries-{self.language_type}/{self.cmd}.bqrs"
        if not os.path.exists(path):
            return

        # analyze ql command
        cmd = f"codeql bqrs decode --format=csv --output={src}/{self.cmd}.csv {path}"
        os.system(cmd)

    @staticmethod
    def load_project_csv(src, cmd):
        out = pd.read_csv(f'{src}/{cmd}.csv')
        return out

    @staticmethod
    def create_dfg_from_csv(csv: pd.DataFrame):
        groups = csv.groupby(['col0', 'col1'])
        context = dict()
        for group_key, group_value in groups:
            variable_name = group_key[0].split()

            words = ";".join(group_value['col2'])
            words = words.split(';')
            words = set(words)
            variable_context = " ".join(words)
            variable_context = process_text(variable_context)

            context[variable_name] = variable_context
        return context


def process_text(text):
    variable_context = text.replace('(...)', '')
    variable_context = variable_context.replace('...', '')
    variable_context = variable_context.replace('=', '')
    return variable_context


def analyze_str(base_path, cmd, skip=True, threads=4):
    """
    run ql for all dataset
    :param base_path: dir path
    :param skip: skip if the dataset had been analyzed
    :param threads:
    :return:
    """
    analyzer = JavaAnalyzer(False)
    analyzer.set_cmd(cmd)
    dirs = tqdm(os.listdir(base_path))

    for proj_dir in dirs:
        dirs.set_description(f"Processing: {proj_dir.ljust(50,' ')}")

        result = analyzer.ql_find_str(f'{base_path}/{proj_dir}', skip=skip, threads=threads)
        if result:
            analyzer.decode_bqrs2csv(f'{base_path}/{proj_dir}')


def decode_bqrs_all(base_path, cmd):
    analyzer = JavaAnalyzer(False)
    analyzer.set_cmd(cmd)
    for proj_dir in tqdm(os.listdir(base_path)):
        analyzer.decode_bqrs2csv(f'{base_path}/{proj_dir}')


def merge_data(base_path, cmd):
    analyzer = JavaAnalyzer(True)
    analyzer.set_cmd('findString')
    out = None
    first = True
    dirs = tqdm(os.listdir(base_path))
    for proj_dir in dirs:
        if not os.path.exists(f'{base_path}/{proj_dir}/{cmd}.csv'):
            continue
        data = Analyzer.load_project_csv(f'{base_path}/{proj_dir}', cmd)
        # add project name
        data['project'] = proj_dir

        if first:
            out = data
            first = False
        else:
            out = pd.concat([out, data], ignore_index=True)
    return out


def analyze_str_context(base_path, str_path, language_type, debug=False):
    ana:Analyzer = None
    if language_type == 'java':
        ana = JavaAnalyzer(debug)
    if language_type == 'cpp':
        ana = CppAnalyzer(debug)
    if language_type == 'python':
        ana = PythonAnalyzer(debug)

    ana.set_cmd('context_to')
    ana.specific_str_context_array(base_path, str_path)

    ana.set_cmd('context_from')
    ana.specific_str_context_array(base_path, str_path)

    decode_bqrs_all(base_path, 'context_to')
    decode_bqrs_all(base_path, 'context_from')
    context_to = merge_data(base_path, cmd='context_to')
    context_from = merge_data(base_path, cmd='context_from')
    context_to.to_csv(f'{base_path}/context_to.csv')
    context_from.to_csv(f'{base_path}/context_from.csv')


class JavaAnalyzer(Analyzer):

    def __init__(self, debug):
        super(JavaAnalyzer, self).__init__(debug)
        self.language_type = "java"


class CppAnalyzer(Analyzer):

    def __init__(self, debug):
        super(CppAnalyzer, self).__init__(debug)
        self.language_type = "cpp"


class PythonAnalyzer(Analyzer):

    def __init__(self, debug):
        super(PythonAnalyzer, self).__init__(debug)
        self.language_type = "python"




