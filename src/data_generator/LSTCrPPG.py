import threading
import cv2
from .PhysNet import PhysNetDataConfig,PhysNetDataGenerator
import numpy as np
import torch.nn.functional as F
from ..common.ppg_interpolat import generate_interpolated_ppg_by_video_capture
from ..common.cache import CacheType, DataSetCache
from ..common.import_tqdm import tqdm
from ..face_detector.mtcnn.detector import detect_faces

class LSTCrPPGDataConfig(PhysNetDataConfig):
    pass

class LSTCrPPGDataGenerator(PhysNetDataGenerator):

    def _normalization(self,X,y):
        # C,T,W,H
        X = X.transpose((3, 0, 1, 2)) / 255
        # y = (y-y.mean())/y.std()
        y = (y - y.min())/(y.max() -y.min())
        return X,y
        image_dirname = self.process_video_path_to_dirname(video_path)
        step = int(self.config.step)
        slice_interval = int(self.config.slice_interval)
        # open video
        video_capture = cv2.VideoCapture(video_path)
        # interpolate
        interpolated_ppg =  generate_interpolated_ppg_by_video_capture(ppg,video_capture)
        ppg_size = len(interpolated_ppg)
        target_height = self.config.width
        target_width = self.config.height
        bvp_queue = list()
        X_queue = list()
        progress_video_bar = tqdm(interpolated_ppg,initial=0, desc=f"Processing videos {video_index+1}") if self.config.print_info else interpolated_ppg
        for i,ppg_strength in enumerate(progress_video_bar):
            ret, frame = video_capture.read()
            if not ret:
                continue
            if i < self.config.discard_front:
                continue
            if i >= ppg_size - self.config.discard_back:
                continue
            X_queue.append(frame)
            bvp_queue.append(ppg_strength)
            if len(bvp_queue) >= slice_interval:
                first_frame = X_queue[0]
                bounding_boxes, landmarks = detect_faces(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
                if len(bounding_boxes) <= 0:
                    X_queue.pop(0)
                    bvp_queue.pop(0)
                    continue
                largest_face = max(bounding_boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
                x1, y1, x2, y2,score = largest_face
                x1, y1, x2, y2 = [int(v) for v in [x1, y1, x2, y2]]
                H,W,C = first_frame.shape
                if x1 >= W or x1 < 0 or y1 >= H or y1 < 0 or \
                    x2 >= W or x2 <= 0 or y2 >= H or y2 <=0 \
                    or (x2 - x1) < 64 or (y2 - y1) < 64:
                    X_queue.pop(0)
                    bvp_queue.pop(0)
                    continue
                bvp_temp = np.array(bvp_queue)
                X_temp = []
                for X in X_queue:
                    X = X[y1:y2,x1:x2,:]
                    try:
                        X = cv2.resize(X, (target_width, target_height), interpolation=cv2.INTER_AREA)
                    except:
                        print(y2-y1)
                    X_temp.append(X)
                X_temp = np.array(X_temp)
                # 归一化
                normalized = self._normalization(X_temp,bvp_temp)
                if normalized is None:
                    bvp_queue.clear()
                    X_queue.clear()
                    continue
                dataset_index = 0
                with self.dataset_index_lock:
                    dataset_index = self.dataset_index
                    self.dataset_index += 1
                cache.save_image(X_temp[0],dataset_index,image_dirname)
                X_temp,bvp_temp = normalized
                cache.save((X_temp,bvp_temp),dataset_index)    
                bvp_queue = bvp_queue[step:]
                X_queue = X_queue[step:]
        if self.config.print_info:
            progress_video_bar.clear()
            progress_video_bar.close()
        pass