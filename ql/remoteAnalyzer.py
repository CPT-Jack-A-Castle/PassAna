# coding: utf-8
import gzip
import http.client
import io
import json
import os

import ssl
import string
import urllib.parse
from urllib.request import urlopen

import requests
from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context


class RemoteAnalyzer(object):
    def __init__(self, bearer='940af9ba7bd1e65d943a80bf40ab6b25808e715cfdf314cb8dc14c561567ff81'):
        self.bearer = bearer

    def get_download(self, project_id, language, file_path):
        try:
            headers = {
                "Authorization": f"Bearer {self.bearer}"}

            url = f'https://lgtm.com/api/v1.0/snapshots/{project_id}/{language}'

            file_size = int(urlopen(url).info().get('Content-Length', -1))

            if os.path.exists(file_path):
                first_byte = os.path.getsize(file_path)  # (3)
            else:
                first_byte = 0
            if first_byte >= file_size: # (4)
                return file_size

            pbar = tqdm(total=file_size, initial=first_byte, unit='B', unit_scale=True, desc=file_path)

            req = requests.get(url, headers=headers, stream=True)

            with open(file_path, 'ab') as f:
                for chunk in req.iter_content(chunk_size=1024):     # (6)
                    if chunk:
                        f.write(chunk)
                        pbar.update(1024)

            pbar.close()
        except Exception as e:
            print(e)

    def get_project(self, project):
        try:
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {self.bearer}"}
            conn = http.client.HTTPSConnection("lgtm.com")
            conn.request('GET', f'/api/v1.0/projects/g/{project}', headers=headers)
            response = conn.getresponse()
            data = json.loads(response.read().decode('utf-8'))
            conn.close()
        except Exception as e:
            pass
            conn.close()

        return data['id']

    def download_dataset(self, filename: str, language, path: str):
        """
        download dataset (zip file) from LGTM
        :param filename: project name like 'linkedin/shaky-android'
        :param language: language that want to get
        :param path: save path
        :return:
        """
        project_id = self.get_project(filename)
        name = filename.split('/')[1]
        self.get_download(project_id, language, f'{path}/{name}_{language}.zip')

if __name__ == '__main__':
    remote = RemoteAnalyzer()
    remote.download_dataset('linkedin/shaky-android', 'java', '/home/rain/program')
