import math
import threading
import numpy as np
import torch
from ...common.cuda_info import get_device
from .get_nets import PNet, RNet, ONet
from .box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from .first_stage import run_first_stage
# LOAD MODELS
pnet = PNet().eval().to(get_device())

rnet = RNet().eval().to(get_device())

onet = ONet().eval().to(get_device())

def detect_faces(frame, min_face_size=20.0,
                 thresholds=[0.6, 0.7, 0.8],
                 nms_thresholds=[0.7, 0.7, 0.7]):
    """
    Arguments:
        image: numpy array.
        min_face_size: a float number.
        thresholds: a list of length 3.
        nms_thresholds: a list of length 3.

    Returns:
        two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
        bounding boxes and facial landmarks.
    """
    frame = np.array(frame,np.float32)
    # BUILD AN IMAGE PYRAMID
    height,width, channel = frame.shape
    min_length = min(height, width)

    min_detection_size = 12
    factor = 0.707  
    # factor = np.sqrt(0.5)

    # scales for scaling the image
    scales = []

    # scales the image so that
    # minimum size that we can detect equals to
    # minimum face size that we want to detect
    m = min_detection_size/min_face_size
    min_length *= m

    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m*factor**factor_count)
        min_length *= factor
        factor_count += 1

    # STAGE 1

    # it will be returned
    bounding_boxes = []
    # run P-Net on different scales
    def process_scales(s):
        boxes = run_first_stage(frame, pnet, scale=s, threshold=thresholds[0])
        bounding_boxes.append(boxes)

    # for s in scales:
    #     process_scales(s)

    cpu_num = 12
    task_groups = np.array_split(scales, math.ceil(len(scales)/cpu_num))
    for tasks in task_groups:
        threads = list()
        for s in tasks:
            thread = threading.Thread(target=process_scales, args=[s])
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

    # collect boxes (and offsets, and scores) from different scales
    bounding_boxes = [i for i in bounding_boxes if i is not None]
    bounding_boxes = np.vstack(bounding_boxes)

    keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
    bounding_boxes = bounding_boxes[keep]

    # use offsets predicted by pnet to transform bounding boxes
    bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
    # shape [n_boxes, 5]

    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 2

    img_boxes = get_image_boxes(bounding_boxes, frame, size=24)
    X_rnet = torch.tensor(img_boxes,device=get_device()).float()
    # X_rnet = torch.FloatTensor(img_boxes)
    # with torch.no_grad():
    output = rnet(X_rnet)
    offsets = output[0].cpu().data.numpy()  # shape [n_boxes, 4]
    probs = output[1].cpu().data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[1])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]

    keep = nms(bounding_boxes, nms_thresholds[1])
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 3
    img_boxes = get_image_boxes(bounding_boxes, frame, size=48)
    if len(img_boxes) == 0:
        return [], []
    X_onet = torch.tensor(img_boxes,device=get_device()).float()
    # X_onet = torch.FloatTensor(img_boxes)
    # with torch.no_grad():
    output = onet(X_onet)
    landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
    offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
    probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds[2])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]
    landmarks = landmarks[keep]

    # compute landmark points
    width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
    height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
    xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
    landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
    landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]

    bounding_boxes = calibrate_box(bounding_boxes, offsets)
    bounding_boxes = convert_to_square(bounding_boxes)
    keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
    bounding_boxes = bounding_boxes[keep]
    landmarks = landmarks[keep]

    return bounding_boxes, landmarks
