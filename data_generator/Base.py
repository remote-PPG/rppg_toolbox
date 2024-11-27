import os
from torch.utils.data import DataLoader
from common.cuda_info import get_device_str,is_cuda_available,set_cuda_index
from common.cache import DataSetCache,CacheDataset,CacheType
class BaseConfig:
    def __init__(self,cache_root:str,cache_type:CacheType,step=20,slice_interval=160,batch_size=1,load_to_memory=False,shuffle=False,num_workers=8,pin_memory=True,print_info=True) -> None:
        self.step = step
        self.slice_interval = slice_interval
        self.batch_size = batch_size
        self.load_to_memory = load_to_memory
        self.cache_root = cache_root
        self.cache_type = cache_type
        self.shuffle = shuffle
        self.num_workers=num_workers
        self.pin_memory=pin_memory
        self.print_info = print_info

class BaseDataGenerator:
    def __init__(self,config:BaseConfig):
        self.config = config
        pass
    def generate(self,data = None):
        config = self.config
        cache = DataSetCache(config.cache_root,config.cache_type)
        if data is not None and config.cache_type == CacheType.NEW_CACHE:
            cache.free()
            self.generate_cache(data,cache)
        if not cache.exist() or cache.size() == 0:
            raise Exception("Error: dataset size is 0, set NEW_CACHE, or check your dataset path")
        dataset = CacheDataset(cache,self.config.load_to_memory)
        if self.config.print_info:
            print(f'dataset size: {len(dataset)}')
        data_loader = DataLoader(dataset, batch_size=config.batch_size,
                                 num_workers=config.num_workers,
                                 pin_memory= config.pin_memory and is_cuda_available(),
                                 pin_memory_device=get_device_str() if is_cuda_available() else "",
                                 shuffle=config.shuffle and len(dataset) > 0)
        return data_loader
    def generate_cache(self,data,cache:DataSetCache):
        raise Exception("请在子类中重写这个函数！")
    
    def process_video_path_to_dirname(self,video_path):
        if video_path.startswith('/'):
            video_path = video_path[1:]
        video_path = video_path.replace('/', '][')
        video_path = os.path.splitext(video_path)[0]
        return f'[{video_path}]'

