import urllib.request
import zipfile
import os

url = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip'
zip_path = 'african-wildlife.zip'

print('Downloading dataset...')
urllib.request.urlretrieve(url, zip_path)

print('Extracting dataset...')
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('.')

os.remove(zip_path)
print('Done.')
