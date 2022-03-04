import time

from ql.remoteAnalyzer import RemoteAnalyzer

if __name__ == '__main__':
    remote = RemoteAnalyzer()
    for line in open('/home/rain/PassAna/ql/python_repo_name.txt', 'r'):
        repo_name = line.replace('\n', '')
        # print(repo_name)
        try:
            remote.download_dataset(repo_name, 'python', '/home/rain/program/python_zip', threshold=500)
        except Exception as e:
            print('analyzer "{}" error as {}'.format(repo_name, e))
        time.sleep(1)
