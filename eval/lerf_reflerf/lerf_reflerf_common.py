import os
import numpy as np
from PIL import Image

LERF_SCENES = {
    "waldo_kitchen": ["frame_00053", "frame_00066", "frame_00089", "frame_00140", "frame_00154"],
    "ramen": ["frame_00006", "frame_00024", "frame_00060", "frame_00065", "frame_00081", "frame_00119", "frame_00128"],
    "figurines": ["frame_00041", "frame_00105", "frame_00152", "frame_00195"],
    "teatime": ["frame_00002", "frame_00025", "frame_00043", "frame_00107", "frame_00129", "frame_00140"]
}

REFLERF_SCENES = {
    "waldo_kitchen": ['frame_00019', 'frame_00035', 'frame_00067', 'frame_00105', 'frame_00162'],
    "ramen": ['frame_00006', 'frame_00024', 'frame_00060', 'frame_00065', 'frame_00081', 'frame_00119', 'frame_00128'],
    "figurines": ['frame_00083', 'frame_00097', 'frame_00146', 'frame_00179'],
    "teatime": ['frame_00002', 'frame_00025', 'frame_00043', 'frame_00107', 'frame_00129', 'frame_00140']
}

def load_image_as_binary(image_path, is_png=False, threshold=10):
    image = Image.open(image_path)
    if is_png:
        image = image.convert('L')
    image_array = np.array(image)
    binary_image = (image_array > threshold).astype(np.int32)
    return binary_image

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:

        return 0.0
    
    return float(intersection / union)

def calculate_accuracy_at_threshold(ious, threshold):
    if len(ious) == 0:
        return 0.0
    
    count = (np.array(ious) > threshold).sum()
    return float(count / len(ious))

def aggregate_metrics(ious):
    if len(ious) == 0:
        return 0.0, 0.0, 0.0
    
    mean_iou = np.mean(ious)
    acc_025 = calculate_accuracy_at_threshold(ious, 0.25)
    acc_050 = calculate_accuracy_at_threshold(ious, 0.5)
    
    return mean_iou, acc_025, acc_050

def get_frame_objects(gt_base, frame_names):
    valid_frames = []
    frame_dict = {}
    
    for frame in frame_names:
        gt_frame_path = os.path.join(gt_base, frame)
        if os.path.exists(gt_frame_path):
            valid_frames.append(frame)

            obj_files = os.listdir(gt_frame_path)
            obj_names = [os.path.splitext(f)[0] for f in obj_files if os.path.isfile(os.path.join(gt_frame_path, f))]
            frame_dict[frame] = obj_names
        else:
            print(f"  warning: GT frame directory missing: {gt_frame_path}")
    
    return frame_dict, valid_frames
