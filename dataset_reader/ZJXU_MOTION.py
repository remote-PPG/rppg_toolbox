import pandas as pd
import numpy as np
from common.import_tqdm import tqdm
import os
import re

class ZJXU_MOTION_Reader():
    def print_root_path(self):
        print(f"Root Path:{self.dataset_path}\nStart Reading Dataset Directory...")
    def __init__(self,dataset_path,dataset = "",dataset_list=[]) -> None:
        self.dataset_path = os.path.expanduser(dataset_path)
        self.dataset_list = dataset_list
        self.dataset = dataset
        pass
    def read(self,show_tqdm=True,print_start=True,print_end=True):
        list_of_video_path = []
        list_of_ppg_data = []
        self.print_root_path()
        progress_bar = tqdm(self.dataset_list, desc="Progress")
        for content_name in progress_bar:
            dataset_path = os.path.join(self.dataset_path,content_name)
            ecg_acp_path = os.path.join(dataset_path, 'ecg.acp')
            ecg_csv_path = os.path.join(dataset_path, 'ecg.csv')
            # 查找名字全是数字的文件夹
            data_dir_path = None
            for item in os.listdir(dataset_path):
                item_path = os.path.join(dataset_path, item)
                if os.path.isdir(item_path) and item.isdigit():  # 确保是数字命名的文件夹
                    data_dir_path = item_path
                    break  # 找到第一个符合条件的文件夹后退出循环
            if data_dir_path is None:
                print(f"warn: no data in [{dataset_path}]")
                continue
            # 查找ppg开头的csv文件
            ppg_data_path = None
            rgb_video_path = None
            for root, dirs, files in os.walk(data_dir_path):
                for file_name in files:
                    if file_name.startswith('ppg') and file_name.endswith('.csv'):
                        ppg_data_path = os.path.join(data_dir_path, file_name)
                    elif file_name.startswith('usb_camera') and file_name.endswith('.avi'):
                        rgb_video_path = os.path.join(data_dir_path, file_name)
            if rgb_video_path is None or ppg_data_path is None:
                print(f"warn: no data in [{data_dir_path}]")
                continue
            if os.path.exists(rgb_video_path) and os.path.exists(ppg_data_path):
                list_of_video_path.append(rgb_video_path)
                list_of_ppg_data.append( self.__ppg_reader(ppg_data_path))
        progress_bar.clear()
        progress_bar.close()
        return list_of_video_path,list_of_ppg_data
    def __ppg_reader(self,path):
        ppg_df = pd.read_csv(path,header=0)
        ppg_data = ppg_df.iloc[:,1].to_numpy(dtype=np.float64)
        return ppg_data
