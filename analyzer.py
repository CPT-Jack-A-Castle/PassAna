# coding: utf-8
import logging
import json
import string

import pexpect
import os


class Analyzer(object):

    def __init__(self, debug=False):
        self._analyze_task = None
        self._ql_task = None
        self.language_type = None

        if debug:
            logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                level=logging.INFO)

    def analyze_create(self, src, external_cmd=None):
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
        # Check the file has been analyzed
        if os.path.exists(f"{src}/{self.language_type}_database "):
            if os.path.exists(f"{src}/{self.language_type}_database/src.zip"):
                return
            else:
                logging.warning('Src have no src.zip')
                raise ValueError('Analyze src at first!')

        # analyze ql command
        cmd = f"codeql database analyze  " \
              f"{src}/{self.language_type}_database ql/{self.language_type}/cmd.ql " \
              f"--format=csv --output={src}/result.csv --rerun --threads {threads}"

        self._ql_task = pexpect.spawn(cmd,timeout=3600)
        while True:
            line = self._ql_task.readline().decode()
            logging.debug(line)
            if line.startswith('Interpreting results'):
                logging.info(f'Interpreting results')
                break
            if line.startswith('A fatal error'):
                logging.error(f'A fatal error')
                break

    def decode_bqrs(self, src, outformat: string = 'json'):
        """
        Decode the bqrs to others
        :param src:
        :param outformat: json, csv
        :return: result
        """
        path = f"{src}/{self.language_type}_database/results/getting-started/codeql-extra-queries-{self.language_type}/cmd.bqrs"
        # analyze ql command
        cmd = f"codeql bqrs decode --format={outformat} --output={src}/out.{outformat} {path}"
        os.system(cmd)

        with open(f'{src}/out.json', 'r') as f:
            out = json.load(f)
        return out


class JavaAnalyzer(Analyzer):

    def __init__(self, debug):
        super(JavaAnalyzer, self).__init__(debug)
        self.language_type = "java"


class CppAnalyzer(Analyzer):

    def __init__(self, debug):
        super(CppAnalyzer, self).__init__(debug)
        self.language_type = "cpp"



if __name__ == '__main__':
    # analyzer = JavaAnalyzer(True)
    # # analyzer.analyze_create('/home/rain/program/java/mall')
    # # analyzer.ql_analyze('/home/rain/program/java/mall')
    # analyzer.decode_bqrs('/home/rain/program/java/mall')
    analyzer = CppAnalyzer(True)
    # analyzer.analyze_create('/home/rain/program/cpp/ffmpeg-3.0')
    # analyzer.ql_analyze('/home/rain/program/cpp/ffmpeg-3.0')
    analyzer.decode_bqrs('/home/rain/program/cpp/ffmpeg-3.0')

