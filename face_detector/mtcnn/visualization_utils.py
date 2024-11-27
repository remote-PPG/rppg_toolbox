import matplotlib.pyplot as plt
import cv2
import numpy as np
def show_bboxes(frame, bounding_boxes, facial_landmarks=[]):
    """Draw bounding boxes and facial landmarks.

    Arguments:
        frame: numpy array RGB
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a float numpy array of shape [n, 10].
    """
    frame = np.copy(frame)
    for b in bounding_boxes:
        x1, y1, x2, y2 = map(lambda v:int(v),b[:-1])
        cv2.rectangle(frame,(x1, y1), (x2, y2),color=(255,255,255))
    for p in facial_landmarks:
        p = list(map(lambda v:int(v),p))
        for i in range(5):
            cv2.circle(frame, 
                center=(int(p[i]), int(p[i+5])),  # 圆心为 (p[0], p[1])
                radius=2,  # 圆的半径
                color=(255, 255, 0),  # 圆的颜色为黄色 (BGR)
                thickness=2)  # 圆的边框厚度
    plt.axis('off')
    plt.imshow(frame)
