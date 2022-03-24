import time

from ql.remoteAnalyzer import RemoteAnalyzer

if __name__ == '__main__':
    remote = RemoteAnalyzer()
    # for line in open('/home/gonghuihui/pwd_proj/PassAna/ql/js_repo_name.txt', 'r'):
    #     repo_name = line.replace('\n', '')
    #
    #     try:
    #         remote.download_dataset(repo_name, 'javascript', '/home/gonghuihui/pwd_proj/js_zip', threshold=100)
    #     except Exception as e:
    #         print('analyzer "{}" error as {}'.format(repo_name, e))
    #     time.sleep(1)
    repo_name = "ElasticEmail/ElasticEmail.WebApiClient-csharp"
    remote.download_dataset(repo_name, 'csharp', '/home/gonghuihui/pwd_proj/csharp_zip', threshold=100)