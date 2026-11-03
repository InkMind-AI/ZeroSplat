import json
import os
import numpy as np
import torch
from plyfile import PlyData

def calculate_metrics_per_class(gt_mask, pred_mask):
    intersection = torch.sum(gt_mask & pred_mask).item()
    union = torch.sum(gt_mask | pred_mask).item()
    correct = intersection
    total = torch.sum(gt_mask).item()
    return intersection, union, correct, total

def load_gt_objects(gt_json_path):
    with open(gt_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    objects = data.get("gt_info", [])
    return data, objects

def _get_vertex_ids_from_labels_ply(ply_path):
    ply_data = PlyData.read(ply_path)
    vd = ply_data["vertex"].data
    if "objectId" in vd.dtype.names:
        return np.asarray(vd["objectId"], dtype=np.int64)
    if "label" in vd.dtype.names:
        return np.asarray(vd["label"], dtype=np.int64)
    raise ValueError(
        f"labels.ply missing objectId or label field: {ply_path}. "
        f"Available columns: {list(vd.dtype.names)}"
    )

def build_gt_mask_from_gt_id(gt_id, labels_ply_path, num_points):
    gt_id_set = {int(x) for x in gt_id if x is not None and int(x) >= 0}
    if not gt_id_set:
        return torch.zeros(num_points, dtype=torch.int64).cuda()
    
    vertex_ids = _get_vertex_ids_from_labels_ply(labels_ply_path)
    n_labels = len(vertex_ids)
    
    if n_labels != num_points:
        raise ValueError(
            f"labels.ply point count ({n_labels}) != point_cloud.ply ({num_points}). "
            f"Ensure both files are from the same scene with aligned counts."
        )
    
    gt_mask = np.isin(vertex_ids, list(gt_id_set)).astype(np.int64)
    return torch.from_numpy(gt_mask).cuda()

def get_point_count_from_ply(ply_path):
    ply_data = PlyData.read(ply_path)
    return len(ply_data["vertex"].data)

def aggregate_scene_metrics(scene_intersections, scene_unions, scene_corrects, scene_totals):
    if not scene_intersections:
        return float("nan"), float("nan"), float("nan")

    scene_ious = []
    for inter, uni in zip(scene_intersections, scene_unions):
        if uni > 0:
            scene_ious.append(inter / uni)

    if not scene_ious:
        return float("nan"), float("nan"), float("nan")

    mean_iou = np.mean(scene_ious)
    total_correct = sum(scene_corrects)
    total_points = sum(scene_totals)
    accuracy = total_correct / total_points if total_points > 0 else float("nan")

    class_accs = []
    for c, t in zip(scene_corrects, scene_totals):
        if t > 0:
            class_accs.append(c / t)
    mean_acc = np.mean(class_accs) if class_accs else float("nan")

    return mean_iou, accuracy, mean_acc
