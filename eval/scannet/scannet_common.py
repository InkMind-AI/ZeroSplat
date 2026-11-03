import os
import numpy as np
import torch
from plyfile import PlyData

NYU40_DICT = {
    0: "unlabeled", 1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair",
    6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf",
    11: "picture", 12: "counter", 13: "blinds", 14: "desk", 15: "shelves",
    16: "curtain", 17: "dresser", 18: "pillow", 19: "mirror", 20: "floormat",
    21: "clothes", 22: "ceiling", 23: "books", 24: "refrigerator", 25: "television",
    26: "paper", 27: "towel", 28: "shower curtain", 29: "box", 30: "whiteboard",
    31: "person", 32: "nightstand", 33: "toilet", 34: "sink", 35: "lamp",
    36: "bathtub", 37: "bag", 38: "otherstructure", 39: "otherfurniture", 40: "otherprop"
}

SCANNET_TARGET_IDS = {
    "19": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36],
    "15": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 33, 34],
    "10": [1, 2, 4, 5, 6, 7, 8, 9, 10, 33],
}

def calculate_metrics_per_class(gt_mask, pred_mask):
    intersection = torch.sum(gt_mask & pred_mask).item()
    union = torch.sum(gt_mask | pred_mask).item()
    correct = intersection
    total = torch.sum(gt_mask).item()
    return intersection, union, correct, total

def read_labels_from_ply(file_path):
    ply_data = PlyData.read(file_path)
    vertex_data = ply_data['vertex'].data
    points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    labels = vertex_data['label']
    return points, labels

def update_gt_labels(labels, target_ids):
    target_id_mapping = {orig: idx + 1 for idx, orig in enumerate(target_ids)}
    updated = np.zeros_like(labels)
    for orig, new in target_id_mapping.items():
        updated[labels == orig] = new
    return updated

def aggregate_scene_metrics(scene_intersections, scene_unions, scene_corrects, scene_totals):
    if not scene_intersections:
        return float('nan'), float('nan'), float('nan')

    scene_ious = []
    for inter, uni in zip(scene_intersections, scene_unions):
        if uni > 0:
            scene_ious.append(inter / uni)

    if not scene_ious:
        return float('nan'), float('nan'), float('nan')

    mean_iou = np.mean(scene_ious)

    total_correct = sum(scene_corrects)
    total_points = sum(scene_totals)
    accuracy = total_correct / total_points if total_points > 0 else float('nan')

    class_accs = []
    for c, t in zip(scene_corrects, scene_totals):
        if t > 0:
            class_accs.append(c / t)
    mean_acc = np.mean(class_accs) if class_accs else float('nan')

    return mean_iou, accuracy, mean_acc
