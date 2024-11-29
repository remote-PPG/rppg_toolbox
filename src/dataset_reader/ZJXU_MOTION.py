from pathlib import Path
import pandas as pd
import numpy as np
from ..common.import_tqdm import tqdm
import os
from typing import List, Tuple, Literal

class PhysiologicalInfo:
    def __init__(self):
        self.bvp_size = 0
        self.spo2_size = 0
        self.pr_size = 0
        self.pi_size = 0
        self.ecg_size = 0
        self.BVP = []
        self.SPO2 = []
        self.PR = []
        self.PI = []
        self.ECG = []
class VideoSet:
    def __init__(self,usb_camera_left:str,usb_camera_right:str,inf_video:str):
        self.usb_camera_left = usb_camera_left
        self.usb_camera_right = usb_camera_right
        self.inf_video = inf_video

class InfoReader:
    @staticmethod
    def read(info_path:str):
        phys_info = PhysiologicalInfo()
        with open(info_path, "rb") as file:
            data = np.frombuffer(file.read(), dtype=np.uint8)
            phys_info.bvp_size = (data[0] << 8) | data[1]
            phys_info.spo2_size = (data[2] << 8) | data[3]
            phys_info.pr_size = (data[4] << 8) | data[5]
            phys_info.pi_size = (data[6] << 8) | data[7]
            phys_info.ecg_size = data[8] << 16 | data[9] << 8 | data[10]
            start_index = 13
            sizes = [phys_info.bvp_size,phys_info.spo2_size,phys_info.pr_size,phys_info.pi_size]
            for i,s in enumerate([phys_info.BVP,phys_info.SPO2,phys_info.PR,phys_info.PI]):
                for j in range(phys_info.bvp_size):
                    s.append(data[start_index+j])
                start_index += sizes[i]
            ecg_data = data[start_index:].view(dtype=np.float32)
            for j in range(phys_info.ecg_size):
                phys_info.ECG.append(ecg_data[j])
        return phys_info




class ZJXU_MOTION_Reader:
    def print_root_path(self):
        print(f"Root Path:{self.dataset_path}\nStart Reading Dataset Directory...")
    def __init__(self,dataset_path,
                 samples: List[str],
                                   scenes: List[Literal['S_L1','S_L2','S_L3','W_L1','W_L2','R_L1','R_L2']]) -> None:
        '''
            @param samples: list of s01~s20
            @param scenes: list of 'S_L1','S_L2','S_L3','W_L1','W_L2','R_L1','R_L2'
        '''
        self.dataset_path = os.path.expanduser(dataset_path)
        self.samples = samples
        self.scenes = scenes
        pass
    def read(self,show_tqdm=True,print_start=True,print_end=True)->Tuple[List[PhysiologicalInfo],List[VideoSet]]:
        list_of_video_path = []
        list_of_info_data = []
        self.print_root_path()
        progress_bar = tqdm(self.samples, desc="Progress")
        for sample in progress_bar:
            for scene in self.scenes:
                dataset_path = os.path.join(self.dataset_path,sample,scene)
                info_path = os.path.join(dataset_path, 'physiological.info')
                usb_camera_left_path = None
                usb_camera_right_path = None
                inf_video_path = None
                for item in os.listdir(dataset_path):
                    item_path = Path(os.path.join(dataset_path, item))
                    if item_path.is_file() and item_path.name.startswith('usb_camera_left') and item_path.name.endswith('.avi'):
                        usb_camera_left_path = str(item_path)
                    if item_path.is_file() and item_path.name.startswith('usb_camera_right') and item_path.name.endswith('.avi'):
                        usb_camera_right_path = str(item_path)
                    if item_path.is_file() and item_path.name.startswith('inf') and item_path.name.endswith('.mp4'):
                        inf_video_path = str(item_path)
                if not os.path.exists(info_path) or usb_camera_left_path is None or usb_camera_right_path is None or inf_video_path is None:
                    print(f"warn: no data in [{dataset_path}]")
                    continue
                list_of_video_path.append(VideoSet(usb_camera_left_path,usb_camera_right_path,inf_video_path))
                list_of_info_data.append(InfoReader.read(info_path))
        progress_bar.clear()
        progress_bar.close()
        return list_of_info_data,list_of_video_path