import torch as __torch
# 检查是否支持 CUDA（GPU 加速）
__cuda_index = 0

def print_info():
    if is_cuda_available():
        print('CUDA:\n---------------------------')
        # 获取当前可用的 CUDA 设备数量
        device_count = __torch.cuda.device_count()
        print(f"PyTorch supports GPU and currently has {device_count} CUDA devices available.")

        # 获取当前默认的 CUDA 设备
        current_device = __torch.cuda.current_device()
        device_name = __torch.cuda.get_device_name(current_device)
        print(f"The current default CUDA device is: {device_name}")
        print('---------------------------')
    else:
        print("PyTorch does not support GPU acceleration.")
def is_cuda_available():
    return __torch.cuda.is_available()
def set_cuda_index(cuda_index = 0):
    global __cuda_index
    __cuda_index = cuda_index
def get_device_str():
    return f"cuda:{__cuda_index}" if is_cuda_available() else "cpu"
def get_device():
    return __torch.device(get_device_str())
