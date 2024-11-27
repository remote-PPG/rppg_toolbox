import torch
from common.filter import detrend, lowpass_filter
from scipy import signal
import numpy as np
from singleton_pattern import load_config

class POS(torch.nn.Module):
    def __init__(self):
        super(POS, self).__init__()
        config = load_config.get_config()
        data_format = config['data_format']
        slice_interval = data_format['slice_interval']
        fs = config['data_format']['fps']
        self.slice_interval = slice_interval
        self.fs = fs
        self.WinSec = 1.0

    '''
        x :(B, T, C, H, W)
    '''
    def forward(self,x):
        # x = torch.permute(x, (0, 2, 1, 3,4))  # =>(B, T, C, H, W)
        x = torch.mean(x, dim=(3, 4))
        x = x.cpu().numpy()
        batch_size, N, num_features = x.shape

        Hs = np.zeros((batch_size, N),np.float32)
        for b in range(batch_size):
            mean_rgbs = x[b]
            l = int(self.fs*self.WinSec)
            ts = mean_rgbs.shape[0]-l
            if ts <= 0 :
                return Hs
            for t in range(0, ts):
                # Step 1: Spatial averaging
                C = mean_rgbs[t:t+l-1,:].T
                #C = mean_rgbs.T
                #Step 2 : Temporal normalization
                mean_color = np.mean(C, axis=1)
                #print("Mean color", mean_color)
                
                diag_mean_color = np.diag(mean_color)
                #print("Diagonal",diag_mean_color)
                
                diag_mean_color_inv = np.linalg.inv(diag_mean_color)
                #print("Inverse",diag_mean_color_inv)
                
                Cn = np.matmul(diag_mean_color_inv,C)
                #Cn = diag_mean_color_inv@C
            
                #Step 3: 
                projection_matrix = np.array([[0,1,-1],[-2,1,1]])
                S = np.matmul(projection_matrix,Cn)
                #S = projection_matrix@Cn

                #Step 4:
                #2D signal to 1D signal
                std = np.array([1,np.std(S[0,:])/np.std(S[1,:])])
                P = np.matmul(std,S)

                #Step 5: Overlap-Adding
                Hs[b,t:t+l-1] = Hs[b,t:t+l-1] +  (P-np.mean(P))/np.std(P)
            data = detrend(Hs[b],100)
            t,data = lowpass_filter(data, fs=self.fs, cutoff_freq=3, order=5)
            Hs[b] = (data - data.min())/(data.max() -data.min())
        BVP = Hs.copy()
        BVP = torch.from_numpy(BVP.copy()).view(batch_size, -1)
        return BVP