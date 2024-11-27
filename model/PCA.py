import torch
from sklearn.decomposition import PCA as skpca
import numpy as np
from common.filter import lowpass_filter

class PCA(torch.nn.Module):
    def __init__(self):
        super(PCA, self).__init__()


    def forward(self,x):
        batch, _, _, _, _ = x.shape
        x = torch.mean(x,dim=(3,4))
        bvp = []
        for i in range(batch):
            X = x[i]
            pca = skpca(n_components=3)
            pca.fit(X.cpu().numpy())
            data = pca.components_[1] * pca.explained_variance_[1]
            t,data = lowpass_filter(data, fs=35.0, cutoff_freq=3, order=5)
            # y_2 = (y_2 - y_2.mean())/y_2.std()
            data = (data - data.min())/(data.max() -data.min())
            bvp.append(torch.from_numpy(data).to(x.device))
        bvp = torch.stack(bvp)
        return bvp