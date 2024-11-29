import pickle
from os import path,listdir,makedirs
import shutil
import cv2
import torch
from torch.utils.data import Dataset
import enum

class CacheType(enum.Enum):
    NEW_CACHE = 1
    READ = 2
class Cache:
    def __init__(self,cache_root,cache_type:CacheType,filename:str) -> None:
        cache_root = path.expanduser(cache_root)
        self.file_path = path.join(cache_root,filename)
        self.image_path = path.join(cache_root,'image')
    def exist(self) -> bool:
        return path.exists(self.file_path) and path.isdir(self.file_path)
    def free(self):
        if self.exist():
            shutil.rmtree(self.file_path)
        if path.exists(self.image_path) and path.isdir(self.image_path):
            shutil.rmtree(self.image_path)
    def size(self) -> int:
        if not self.exist():
            return 0
        subdirectories = [d for d in listdir(self.file_path) if path.isfile(path.join(self.file_path, d))]
        return len(subdirectories)
    def save_image(self,frame,image_name,dirname="default"):
        save_dirpath = path.join(self.image_path,dirname)
        try:
            makedirs(save_dirpath, exist_ok=True)
            cv2.imwrite(path.join(save_dirpath,f'{image_name}.jpg'), frame)
        except Exception as e:
            print(f"Error saving Image, suffix : {dirname}/{image_name} , {e}")
class DataSetCache(Cache):
    def __init__(self, cache_root, cache_type: CacheType) -> None:
        super().__init__(cache_root, cache_type, 'data_set')
    def save(self,data,suffix):
        try:
            makedirs(self.file_path, exist_ok=True)
            with open(path.join(self.file_path,f'data_{str(suffix)}.pkl'), 'wb') as file:
                pickle.dump(data, file)
                file.close()
        except Exception as e:
            print(f"Error saving DataLoader, suffix : {suffix} , {e}")
    def read(self,suffix):
        with open(path.join(self.file_path,f'data_{str(suffix)}.pkl'), 'rb') as file:
            data = pickle.load(file)
            file.close()
        return data
class ModelCache(Cache):
    def __init__(self, cache_root, cache_type: CacheType) -> None:
        super().__init__(cache_root, cache_type, 'model')
    def save_model(self,model:torch.nn.Module):
        with open(path.join(self.file_path,self.model_name), 'wb') as file:
            pickle.dump(model, file)
            file.close()
    def read_model(self)->torch.nn.Module:
        model_path = path.join(self.file_path,self.model_name)
        if not path.exists(model_path):
            return None
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            file.close()
        return model
class CacheDataset(Dataset):
    def __init__(self, cache:DataSetCache,load_to_memory=False):
        self.cache = cache
        self.tensor_data_list_cache = [None] * self.__len__()
        self.load_to_memory = load_to_memory
    def __len__(self):
        return self.cache.size()
    def __getitem__(self, index):
        if self.load_to_memory and self.tensor_data_list_cache[index] is not None:
            return self.tensor_data_list_cache[index]
        data_set = list(self.cache.read(index))
        for i,data in enumerate(data_set):
            data_set[i] = torch.tensor(data).float()
        if self.load_to_memory:
            self.tensor_data_list_cache[index] = data_set
        return data_set
