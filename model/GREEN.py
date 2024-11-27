import torch
from singleton_pattern import load_config
from common.filter import lowpass_filter

class GREEN(torch.nn.Module):
    def __init__(self):
        super(GREEN, self).__init__()
        config = load_config.get_config()
        data_format = config['data_format']
        slice_interval = data_format['slice_interval']
        fs = config['data_format']['fps']
        self.slice_interval = slice_interval
        self.fs = fs
    def forward(self,x):
        [batch, length] = x.shape
        data = x.data.cpu().numpy()
        for b in range(batch):
            t,cal = lowpass_filter(data[b], fs=self.fs, cutoff_freq=3.3, order=5)
            data[b][:] = cal
        x.data = torch.tensor(data,device=x.device)
        return x

    def train_model(self,dataloader):
        self.train()
        for batch_X, batch_y in dataloader:
            pass
        self.eval()
