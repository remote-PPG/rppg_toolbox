import torch
import torch.nn as nn
import numpy as np
from scipy.signal import welch
from scipy.signal import find_peaks

class PeakRefineHR(torch.nn.Module):
    def __init__(self,fps=30,confidence = (0.8,0.7,0.6),hr_min=40,hr_max=250):
        super(PeakRefineHR, self).__init__()
        self.hr_before = (hr_min + hr_max) * 0.4
        self.fps = fps
        self.confidence = sorted(confidence,reverse=True)
        self.hr_min = hr_min
        self.hr_max = hr_max
    def forward(self,x):
        x = x.to('cpu').numpy()
        hr_list = []
        for X in x:
            hr = self.hr_before
            for confidence in self.confidence:
                peak_hr = calculate_peak_hr(X,self.fps,confidence)
                if peak_hr > self.hr_min and peak_hr < self.hr_max:
                    hr = peak_hr
                    self.hr_before = peak_hr
                    break
            hr_list.append(hr)
        return torch.tensor(hr_list)


class PSD_HR(nn.Module):
    def __init__(self,fps,hr_min = 40, hr_max = 250):
        super(PSD_HR, self).__init__()
        self.fps = fps
        self.hr_min = hr_min
        self.hr_max = hr_max
        self.before_hr = round((hr_min+hr_max) * 0.4,2) 
    def forward(self, ppg_signal_batch):
        # ppg_signal_batch: input PPG signals (batch_size, signal_length)
        # fps: frames per second (sampling rate)
        batch_size = ppg_signal_batch.shape[0]
        heart_rates = []
        for i in range(batch_size):
            ppg_signal = ppg_signal_batch[i, :]
            ppg_signal_np = ppg_signal.cpu().detach().numpy()
            heart_rate_bpm = psd_hr(ppg_signal_np,self.fps)
            if heart_rate_bpm <= self.hr_min or heart_rate_bpm >= self.hr_max:
                heart_rate_bpm = self.before_hr
            heart_rates.append(heart_rate_bpm)
            self.before_hr = heart_rate_bpm
        heart_rates_tensor = torch.tensor(heart_rates)
        return heart_rates_tensor



def calculate_peak_hr(x,fps,confidence=0.8):
    x_len = len(x)
    peaks, _ = find_peaks(x,prominence=0.4)
    peaks_space_list = []
    peak_index_before = None
    for peak_index in peaks:
        if peak_index_before is None:
            if x[peak_index] > confidence:
                peak_index_before = peak_index
            continue
        d = peak_index - peak_index_before
        if x[peak_index] <= confidence:
            peak_index_before = None
            continue
        peaks_space_list.append(d)
        peak_index_before = peak_index
    p_len = len(peaks_space_list)
    if p_len > 3:
        half_mean_peaks_space = np.array(peaks_space_list).mean() / 2
        if peaks_space_list[0] < half_mean_peaks_space:
            peaks_space_list.pop(0)
        if (x_len - peaks_space_list[-1]) < half_mean_peaks_space:
            peaks_space_list.pop(-1)
        peaks_space_array = np.array(peaks_space_list)
        median = np.median(peaks_space_array)
        mean = peaks_space_array.mean()
        hr = fps / (mean) * 60
        return round(hr,2)
    return 0

def psd_hr(ppg_signal, fps):
    nperseg = min(256, len(ppg_signal))
    nfft = 40 * len(ppg_signal)
    frequencies, psd = welch(ppg_signal, fps, nperseg=nperseg,nfft=nfft)
    peak_frequency = frequencies[np.argmax(psd)]
    heart_rate_bpm = peak_frequency * 60
    return round(heart_rate_bpm,2)
