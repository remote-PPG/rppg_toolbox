import threading
import cv2
from .Base import BaseDataGenerator as Base,BaseConfig
import numpy as np
import torch.nn.functional as F
from ..common.ppg_interpolat import generate_interpolated_ppg_by_video_capture
from ..common.cache import CacheType, DataSetCache
from ..common.import_tqdm import tqdm

class DeepPhysDataConfig(BaseConfig):
    def __init__(self, cache_root: str, cache_type: CacheType, step=20, slice_interval=160, batch_size=1, load_to_memory=False, shuffle=False, num_workers=8, pin_memory=True, print_info=True,\
                 width=128,height=128,generate_num_workers=1,discard_front = 0,discard_back = 0) -> None:
        super().__init__(cache_root, cache_type, step, slice_interval, batch_size, load_to_memory, shuffle, num_workers, pin_memory, print_info)
        self.width=width
        self.height=height
        self.generate_num_workers = generate_num_workers
        self.discard_front = discard_front
        self.discard_back = discard_back
    pass

class DeepPhysDataGenerator(Base):
    def __init__(self, config:DeepPhysDataConfig):
        super().__init__(config)
    def generate_cache(self,data,cache:DataSetCache):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.dataset_index_lock = threading.Lock()
        self.dataset_index = 0

        video_paths,ppgs = data
        if self.config.print_info:
            print(f"Start Generator Data...")

        progress_bar = tqdm(video_paths, desc="Progress") if self.config.print_info else video_paths

        progress_threads = []
        for video_index,video_path in enumerate(progress_bar):
            if len(progress_threads) < self.config.generate_num_workers:
                thread = threading.Thread(target=self._process_video, args=(cache,video_path,ppgs[video_index],video_index))
                progress_threads.append(thread)
                thread.start()
            # waiting
            while len(progress_threads) >= self.config.generate_num_workers or \
                (len(progress_threads) != 0 and len(video_paths) == video_index + 1):
                for thread in progress_threads:
                    thread.join(0.2)
                progress_threads = list(filter(lambda p:p.is_alive(),progress_threads))

    def _process_video(self,cache,video_path,ppg,video_index):
        step = int(self.config.step)
        slice_interval = int(self.config.slice_interval)
        # open video
        video_capture = cv2.VideoCapture(video_path)
        # interpolate
        interpolated_ppg =  generate_interpolated_ppg_by_video_capture(ppg,video_capture)
        ppg_size = len(interpolated_ppg)
        y_queue = list()
        factor_queue = list()
        progress_video_bar = tqdm(interpolated_ppg,initial=0, desc=f"Processing videos {video_index+1}") if self.config.print_info else interpolated_ppg
        face_x,face_y,face_h,face_w = 0,0,0,0
        for i,ppg_strength in enumerate(progress_video_bar):
            ret, frame = video_capture.read()
            if not ret or i < self.config.discard_front:
                continue
            if i >= ppg_size - self.config.discard_back:
                break
            if face_h == 0 and face_w == 0 and face_x == 0 and face_y == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(faces) > 0:
                    largest_face = max(faces, key=lambda box: box[2] * box[3])
                    face_x, face_y, face_w, face_h = largest_face
                else:
                    # print("no-face")
                    continue
            frame = frame[face_y:face_y+face_h,face_x:face_x+face_w,:]
            target_height = self.config.width
            target_width = self.config.height
            frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            frame_resized = frame_resized / 255.0
            factor_queue.append(frame_resized)
            y_queue.append(ppg_strength)
            if len(y_queue) >= slice_interval:
                y_temp = np.array(y_queue)
                factor_temp = np.array(factor_queue)
                factor_temp,y_temp = self._normalization(factor_temp,y_temp)
                dataset_index = 0
                with self.dataset_index_lock:
                    dataset_index = self.dataset_index
                    self.dataset_index += 1
                cache.save((factor_temp,y_temp),dataset_index)    
                y_queue = y_queue[step:]
                factor_queue = factor_queue[step:]
        if self.config.print_info:
            progress_video_bar.clear()
            progress_video_bar.close()
        pass

    def _normalization(self,X,y):
        X = X.transpose((0, 3, 2, 1))
        [T,C,W,H] = X.shape

        frame_diff = np.diff(X, axis=1)
        frame_sum = X[:, 1:] + X[:, :-1]
        epsilon = 1e-8
        frame_result = frame_diff / (frame_sum + epsilon)
        motion_input = np.zeros((T, C, W, H))
        motion_input[:, 1:] = frame_result

        # y = (y-y.mean())/y.std()
        y_diff = np.diff(y)
        y_true = np.zeros_like(y)
        y_true[1:] = y_diff
        # y = (y - y.min())/(y.max() -y.min())
        return np.array([X,motion_input]),y_true
