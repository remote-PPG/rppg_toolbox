import torch
import torch.nn.functional as F
from torch import nn

class GaussianBlur3D(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0):
        super(GaussianBlur3D, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        kernel = self.get_gaussian_kernel3d(kernel_size, sigma)
        self.register_buffer('gaussian_kernel', kernel)
        
        # 将2D高斯核扩展为3D卷积核，其中时间维度T的卷积核为1
        self.gaussian_kernel = kernel.view(1, 1, 1, kernel_size, kernel_size)
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        # 在C维度上进行卷积，groups=C表示对每个通道独立卷积
        gaussian_kernel = self.gaussian_kernel.expand(C, 1, 1, self.kernel_size, self.kernel_size)
        padding = (0, self.kernel_size // 2, self.kernel_size // 2)  # 对H, W维度进行适当填充
        
        # 使用3D卷积，其中T方向的卷积核为1，不会影响时间维度
        x_blurred = F.conv3d(x, gaussian_kernel, padding=padding, groups=C)
        return x_blurred


    @staticmethod
    def get_gaussian_kernel3d(kernel_size=5, sigma=1.0):
        """
            生成3D的高斯卷积核，其中时间维度T为1
        """
        def gaussian_1d(size, sigma):
            coords = torch.arange(size).float() - (size - 1) / 2.0
            g = torch.exp(-0.5 * (coords / sigma).pow(2))
            return g / g.sum()
        kernel_1d = gaussian_1d(kernel_size, sigma)
        kernel_3d = kernel_1d.view(1, -1, 1) @ kernel_1d.view(-1, 1, 1)
        kernel_3d = kernel_3d.unsqueeze(0)
        
        return kernel_3d
