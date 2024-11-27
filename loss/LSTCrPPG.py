import torch
import torch.fft as fft
from torch import nn


class LSTCrPPGLoss(nn.Module):
    def __init__(self):
        super(LSTCrPPGLoss, self).__init__()
        self.timeLoss = nn.MSELoss()
        self.lambda_value = 0.2
        self.alpha = 1.0
        self.beta = 0.5

    def forward(self, predictions, targets):
        if len(predictions.shape) == 1:
            predictions = predictions.view(1, -1)
        if len(targets.shape) == 1:
            targets = targets.view(1, -1)

        # predictions = (predictions - torch.mean(predictions)) / torch.std(predictions)
        # targets = (targets - torch.mean(targets)) / torch.std(targets)

        targets = torch.nn.functional.normalize(targets, dim=1)
        predictions = torch.nn.functional.normalize(predictions, dim=1)

        l_time = self.timeLoss(predictions, targets)
        l_frequency = self.frequencyLoss(predictions, targets)
        return self.alpha * l_time + self.beta * l_frequency

    def frequencyLoss(self, predictions, target):
        batch, n = predictions.shape
        predictions = self.calculate_rppg_psd(predictions)
        target = self.calculate_rppg_psd(target)
        di = torch.log(predictions) - torch.log(target)
        sum_di_squared = torch.sum(di ** 2, dim=-1)
        sum_di = torch.sum(di, dim=-1)

        hybrid_loss = (1 / n) * sum_di_squared - (self.lambda_value / (n ** 2)) * sum_di ** 2
        loss = torch.sum(hybrid_loss) / batch
        return loss

    def calculate_rppg_psd(self, rppg_signal):
        spectrum = fft.fft(rppg_signal)
        # 복소 곱을 사용하여 PSD 계산
        psd = torch.abs(spectrum) ** 2

        return psd


