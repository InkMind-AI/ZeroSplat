#!/usr/bin/env python3
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
from plyfile import PlyData
from tqdm import tqdm

from scannet.scannet_common import (
    NYU40_DICT,
    SCANNET_TARGET_IDS,
    calculate_metrics_per_class,
    read_labels_from_ply,
    update_gt_labels,
    aggregate_scene_metrics,
)
from scannet.scannet_vis import save_semantic_slices

DEFAULT_SCENE_LIST = [
    'scene0000_00', 'scene0062_00', 'scene0070_00', 'scene0097_00', 'scene0140_00',
    'scene0200_00', 'scene0347_00', 'scene0400_00', 'scene0590_00', 'scene0645_00'
]

def run_scene_eval(
    scan_name,
    data_root,
    iteration,
    use_knn,
    k,
    threshold,
    target_ids,
    save_vis,
    verbose,
):
    model_path = os.path.join(data_root, scan_name)
    gt_file = os.path.join(model_path, f"{scan_name}_vh_clean_2.labels.ply")
    ply_path = os.path.join(model_path, f"point_cloud/iteration_{iteration}/point_cloud.ply")

    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"GT file not found: {gt_file}")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Point cloud not found: {ply_path}")

    points_gt, labels_gt = read_labels_from_ply(gt_file)
    target_names = [NYU40_DICT[i] for i in target_ids]
    updated_gt_labels = torch.from_numpy(update_gt_labels(labels_gt, target_ids).astype(np.int64)).cuda()

    ply_data = PlyData.read(ply_path)
    vertex_data = ply_data['vertex'].data
    model_points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T

    if len(points_gt) != len(model_points):
        raise ValueError(
            f"GT point count ({len(points_gt)}) != model point count ({len(model_points)}); "
            "ensure point clouds share the same source."
        )
    coord_diff = np.linalg.norm(points_gt[:5] - model_points[:5], axis=1)
    if np.any(coord_diff > 1e-3):
        if verbose:
            print(f"warning: {scan_name}: coordinate mismatch; metrics may be inaccurate")

    scene_intersections = []
    scene_unions = []
    scene_corrects = []
    scene_totals = []
    scene_ious = []

    for idx, target_cls_id in enumerate(range(1, len(target_names) + 1)):
        target_name_str = target_names[idx]
        gt_mask = (updated_gt_labels == target_cls_id)
        gt_cnt = torch.sum(gt_mask).item()
        if gt_cnt == 0:
            continue

        if use_knn:
            label_path = os.path.join(model_path, "knn_label", f"{target_name_str}_fixed_labels_k{k}_t{threshold}.pth")
        else:
            label_path = os.path.join(
                model_path, "zero_out", target_name_str,
                "train", f"ours_{iteration}", "label_refinement", "refined_labels.pth"
            )

        if not os.path.exists(label_path):
            if verbose:
                print(f"  warning: missing label file {os.path.basename(label_path)}; IoU=0")
            inter, uni, corr, tot = 0, gt_cnt, 0, gt_cnt
            cls_iou = 0.0
            pred_cnt = 0
        else:
            data = torch.load(label_path, map_location="cuda")
            pred_labels = data['refined_label']
            pred_mask = (pred_labels == 1)
            if pred_mask.dim() > 1:
                pred_mask = pred_mask.squeeze()
            if len(pred_mask) != len(gt_mask):
                raise ValueError(
                    f"class {target_name_str}: prediction count ({len(pred_mask)}) != GT ({len(gt_mask)})"
                )
            pred_mask = pred_mask.to(gt_mask.device)
            inter, uni, corr, tot = calculate_metrics_per_class(gt_mask, pred_mask)
            cls_iou = inter / uni if uni > 0 else 0.0
            pred_cnt = pred_mask.sum().item()

            if save_vis:
                save_semantic_slices(
                    model_points, pred_mask, target_name_str, model_path,
                    subdir="knn" if use_knn else "",
                )

        scene_intersections.append(inter)
        scene_unions.append(uni)
        scene_corrects.append(corr)
        scene_totals.append(tot)
        scene_ious.append(cls_iou)

        if verbose:
            ratio = pred_cnt / gt_cnt if gt_cnt > 0 else 0
            print(f"  {target_name_str:<15} | GT:{gt_cnt:<6} | Pred:{pred_cnt:<6} | IoU:{cls_iou:.4f}")

    mean_iou, accuracy, mean_acc = aggregate_scene_metrics(
        scene_intersections, scene_unions, scene_corrects, scene_totals
    )
    return mean_iou, accuracy, mean_acc, scene_ious

def main():
    parser = argparse.ArgumentParser(description="ScanNet 3D semantic segmentation evaluation")
    parser.add_argument("--data_root", type=str, required=True,
                        help="ScanNet data root")
    parser.add_argument("--scene_list", type=str, nargs="+", default=None,
                        help="Scene list; default built-in subset")
    parser.add_argument("--iteration", type=int, default=90000)
    parser.add_argument("--use_knn", action="store_true", help="Use KNN-densified labels")
    parser.add_argument("--k", type=int, default=40, help="KNN k (when --use_knn)")
    parser.add_argument("--threshold", type=float, default=0.8, help="KNN threshold (when --use_knn)")
    parser.add_argument("--num_classes", type=str, default="19", choices=["10", "15", "19"],
                        help="Number of evaluation classes")
    parser.add_argument("--save_vis", type=bool,default=True, help="Save visualizations")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose logging")

    args = parser.parse_args()
    scene_list = args.scene_list or DEFAULT_SCENE_LIST
    target_ids = SCANNET_TARGET_IDS[args.num_classes]

    all_mious, all_accs, all_maccs = [], [], []
    for scan_name in tqdm(scene_list, desc="Scenes"):
        try:
            miou, acc, macc, _ = run_scene_eval(
                scan_name=scan_name,
                data_root=args.data_root,
                iteration=args.iteration,
                use_knn=args.use_knn,
                k=args.k,
                threshold=args.threshold,
                target_ids=target_ids,
                save_vis=args.save_vis,
                verbose=args.verbose,
            )
            all_mious.append(miou)
            all_accs.append(acc)
            all_maccs.append(macc)
            if args.verbose:
                print(f"  {scan_name} | mIoU: {miou:.4f} | Acc: {acc:.4f} | mAcc: {macc:.4f}")
        except Exception as e:
            print(f"  {scan_name} evaluation failed: {e}")
            raise

    if all_mious:
        print("\n" + "=" * 50)
        print("ScanNet evaluation summary")
        print("=" * 50)
        print(f"  mIoU:  {np.nanmean(all_mious):.4f}")
        print(f"  Acc:   {np.nanmean(all_accs):.4f}")
        print(f"  mAcc:  {np.nanmean(all_maccs):.4f}")
        print("=" * 50)

if __name__ == "__main__":
    main()
