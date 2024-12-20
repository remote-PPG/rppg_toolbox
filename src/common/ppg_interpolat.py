import cv2 as __cv2
import numpy as np
def generate_interpolated_ppg_by_video_capture(ppg,video_capture):
    ppg_array = np.array(ppg)
    # Get video frame rate
    fps = video_capture.get(__cv2.CAP_PROP_FPS)
    # 获取视频总帧数
    video_frames = int(video_capture.get(__cv2.CAP_PROP_FRAME_COUNT))
    # print(fps,video_frames)
    if fps == 0 or video_frames == 0:
        raise Exception("This video has a wrong type")
    # 计算视频的总时长（单位：秒）
    total_duration = video_frames / fps
    # ppg值
    frame_time_stamps = np.linspace(0,total_duration,video_frames)
    frame_time_stamps_xp = np.linspace(0,total_duration,ppg_array.shape[0])
    interpolated_ppg = np.interp(
        frame_time_stamps,
        frame_time_stamps_xp,
        ppg_array
        )
    return interpolated_ppg
def generate_interpolated_ppg(ppg_array,video_path):
    cap = __cv2.VideoCapture(video_path)
    try:
        return generate_interpolated_ppg_by_video_capture(ppg_array,cap)
    except:
        raise Exception(f"This video has a wrong type:{video_path}")