import time

from ql.remoteAnalyzer import RemoteAnalyzer

if __name__ == '__main__':
    remote = RemoteAnalyzer()
    for line in open('/home/rain/PassAna/ql/python_repo_name.txt', 'r'):
        repo_name = line.replace('\n', '')
        # print(repo_name)
        try:
            remote.download_dataset(repo_name, 'python', '/home/rain/program/python_zip')
        except:
            print('analyzer "{}" not found!'.format(repo_name))
        time.sleep(1)
