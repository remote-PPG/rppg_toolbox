'''
  Adapted from here: https://github.com/ZitongYu/PhysFormer/TorchLossComputer.py
  Modifed based on the HR-CNN here: https://github.com/radimspetlik/hr-cnn
'''
import torch
import torch.nn as nn
from .PhysFormer.TorchLossComputer import TorchLossComputer
from .PhysNet import Neg_PearsonLoss

class PhythmFormer_Loss(nn.Module):
    def __init__(self,FPS,diff_flag=1):
        self.FPS = FPS
        self.diff_flag = diff_flag
        super(PhythmFormer_Loss,self).__init__()
        self.criterion_Pearson = Neg_PearsonLoss()
    def forward(self, pred_ppg, labels):
        loss_time = self.criterion_Pearson(pred_ppg.view(1,-1) , labels.view(1,-1))
        loss_CE , loss_distribution_kl = TorchLossComputer.Frequency_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1), diff_flag=self.diff_flag, Fs=self.FPS, std=3.0)
        loss_hr = TorchLossComputer.HR_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1), diff_flag=self.diff_flag, Fs=self.FPS, std=3.0)
        if torch.isnan(loss_time) :
           loss_time = 0

        loss = 0.2 * loss_time + 1.0 * loss_CE + 1.0 * loss_hr
        return loss








