# coding: utf-8
import logging
import json
import string

import pexpect
import os
import csv
import pandas as pd


class Analyzer(object):

    def __init__(self, debug=False):
        # Analyze task handle
        self._analyze_task = None
        # Ql select task handle
        self._ql_task = None
        # Language Type
        self.language_type = None

        if debug:
            logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                level=logging.INFO)

    def analyze_create(self, src, external_cmd=None):
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

    def ql_analyze(self, src, threads=1):
        """
        Run Ql file to analyze the database
        :param src: file path
        :param threads: thread for running (default=1)
        """

        # Check the file has been analyzed
        if os.path.exists(f"{src}/{self.language_type}_database "):
            if os.path.exists(f"{src}/{self.language_type}_database/src.zip"):
                return
            else:
                logging.error('Src have no src.zip')
                raise ValueError('Analyze src at first!')

        # analyze ql command
        cmd = f"codeql database analyze  " \
              f"{src}/{self.language_type}_database ql/{self.language_type}/cmd.ql " \
              f"--format=csv --output={src}/result.csv --rerun --threads {threads}"

        self._ql_task = pexpect.spawn(cmd, timeout=3600)
        while True:
            line = self._ql_task.readline().decode()
            logging.debug(line)
            if line.startswith('Interpreting results'):
                logging.info(f'Interpreting results')
                break
            if line.startswith('A fatal error'):
                logging.error(f'A fatal error')
                break

    def decode_bqrs2json(self, src):
        """
        Decode the .bqrs file to json
        :param src:
        :return: result
        """
        path = f"{src}/{self.language_type}_database/results/getting-started/codeql-extra-queries-{self.language_type}/cmd.bqrs"
        # analyze ql command
        cmd = f"codeql bqrs decode --format=json --output={src}/out.json {path}"
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
        path = f"{src}/{self.language_type}_database/results/" \
               f"getting-started/codeql-extra-queries-{self.language_type}/cmd.bqrs"
        # analyze ql command
        cmd = f"codeql bqrs decode --format=csv --output={src}/out.csv {path}"
        os.system(cmd)

    @staticmethod
    def load_project_csv(src):
        out = pd.read_csv(f'{src}/out.csv')
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



