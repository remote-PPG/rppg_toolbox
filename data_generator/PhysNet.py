import threading
import cv2
from data_generator.Base import BaseDataGenerator as Base,BaseConfig
import numpy as np
import torch.nn.functional as F
from common.ppg_interpolat import generate_interpolated_ppg_by_video_capture
from common.cache import CacheType, DataSetCache
from common.import_tqdm import tqdm
from face_detector.mtcnn.detector import detect_faces

class PhysNetDataConfig(BaseConfig):
    def __init__(self, cache_root: str, cache_type: CacheType, step=20, slice_interval=160, batch_size=1, load_to_memory=False, shuffle=False, num_workers=8, pin_memory=True, print_info=True,\
                 width=128,height=128,generate_num_workers=1,discard_front = 0,discard_back = 0) -> None:
        super().__init__(cache_root, cache_type, step, slice_interval, batch_size, load_to_memory, shuffle, num_workers, pin_memory, print_info)
        self.width=width
        self.height=height
        self.generate_num_workers = generate_num_workers
        self.discard_front = discard_front
        self.discard_back = discard_back
    pass

class PhysNetDataGenerator(Base):
    def __init__(self, config:PhysNetDataConfig):
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

    def _normalization(self,X,y):
        # C,T,W,H
        X = X.transpose((3, 0, 1, 2)) / 255
        y = (y-y.mean())/y.std()
        # y = (y - y.min())/(y.max() -y.min())
        return X,y
    
    def _process_video(self,cache:DataSetCache,video_path,ppg,video_index):
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
                # 第一帧人脸,识别区域
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
                # gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
                # faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
                # if len(faces) <= 0:
                #     X_queue.pop(0)
                #     bvp_queue.pop(0)
                #     continue
                # largest_face = max(faces, key=lambda box: box[2] * box[3])
                # x, y, w, h = largest_face
                # 裁剪和缩放
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