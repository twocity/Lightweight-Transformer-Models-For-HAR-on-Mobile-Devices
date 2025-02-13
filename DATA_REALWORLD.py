import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
import json

# 定义数据集 URL 和本地存储路径
DATASET_URL = "https://sensor.informatik.uni-mannheim.de/~muaddi/dataset.zip"
data_directory = "realworld2016_dataset.zip"

def download_dataset():
    """下载数据集"""
    if not os.path.exists(data_directory):
        print("downloading...")
        urllib.request.urlretrieve(DATASET_URL, data_directory)
    else:
        print("dataset already downloaded")

def extract_dataset():
    """解压数据集"""
    print("extracting data")
    extract_dir = "realworld_dataset"
    if not os.path.exists(extract_dir):
        # 确保下载的文件是完整的
        if os.path.getsize(data_directory) > 0:
            try:
                with zipfile.ZipFile(data_directory, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            except zipfile.BadZipFile:
                print("Error: Downloaded file is corrupted or not a valid zip file")
                # 删除损坏的文件
                os.remove(data_directory)
                print("Corrupted file removed. Please run the script again to re-download")
                return False
        else:
            print("Error: Downloaded file is empty")
            return False
    return True

def main():
    # 1. 下载数据集
    download_dataset()
    
    # 2. 检查文件完整性
    if not os.path.exists(data_directory):
        print("Error: Dataset file not found")
        return
    
    # 3. 解压数据集
    if not extract_dataset():
        return

    # 4. 处理数据
    print("Processing data...")
    # ... 后续数据处理代码 ...

if __name__ == "__main__":
    main() 