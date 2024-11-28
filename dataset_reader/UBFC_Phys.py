import pandas as pd
import numpy as np
from common.import_tqdm import tqdm
import os
import re
from typing import List, Tuple

class UBFCPhysDatasetReader():
    def print_root_path(self):
        print(f"Root Path:{self.dataset_path}:T{self.dataset}\nStart Reading Dataset Directory...")
    def __init__(self,dataset_path,dataset = 1,dataset_list=[]) -> None:
        self.dataset_path = os.path.expanduser(dataset_path)
        self.dataset = dataset
        self.dataset_list = dataset_list
        pass
    def read(self,show_tqdm=True,print_start=True,print_end=True) -> Tuple[List[str],List[List[float]]]:
        list_of_video_path = []
        list_of_ppg_data = []
        self.print_root_path()
        progress_bar = tqdm(self.dataset_list, desc="Progress")
        for content_name in progress_bar:
            ppg_data_path = os.path.join(self.dataset_path,content_name, f'bvp_{content_name}_T{self.dataset}.csv')  # PPG文件
            rgb_video_path = os.path.join(self.dataset_path,content_name, f'vid_{content_name}_T{self.dataset}.avi')
            if os.path.exists(rgb_video_path) and os.path.exists(ppg_data_path):
                list_of_video_path.append(rgb_video_path)
                list_of_ppg_data.append( self.__ppg_reader(ppg_data_path))
        progress_bar.clear()
        progress_bar.close()
        return list_of_video_path,list_of_ppg_data
    def __ppg_reader(self,path):
        ppg_df = pd.read_csv(path,header=None)
        ppg_data = ppg_df.iloc[:,0].to_numpy(dtype=np.float64)
        return ppg_data
