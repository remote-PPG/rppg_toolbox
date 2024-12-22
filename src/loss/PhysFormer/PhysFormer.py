from .TorchLossComputer import TorchLossComputer
import torch
import torch.nn as nn
from ..PhysNet import Neg_PearsonLoss
import math

class PhysFormer_Loss(nn.Module): 
    def __init__(self):
        super(PhysFormer_Loss,self).__init__()
        self.criterion_Pearson = Neg_PearsonLoss()

    def forward(self, pred_ppg, labels , epoch , FS , diff_flag):       
        loss_rPPG = self.criterion_Pearson(pred_ppg.view(1,-1) , labels.view(1,-1))
        loss_CE , loss_distribution_kl = TorchLossComputer.Frequency_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1) , diff_flag = diff_flag , Fs = FS, std=1.0)
        if torch.isnan(loss_rPPG) : 
           loss_rPPG = 0
        if epoch >30:
            a = 1.0
            b = 5.0
        else:
            a = 1.0
            b = 1.0*math.pow(5.0, epoch/30.0)

        loss = a * loss_rPPG + b * (loss_distribution_kl + loss_CE)
        return loss
    