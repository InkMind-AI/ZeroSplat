#!/usr/bin/env python3
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
from plyfile import PlyData
from tqdm import tqdm

from eval.gr_scannet.gr_scannet_common import (
    load_gt_objects,
    build_gt_mask_from_gt_id,
    calculate_metrics_per_class,
    aggregate_scene_metrics,
)
from eval.gr_scannet.gr_scannet_vis import save_semantic_slices
from preprocess.read_reftext import resolve_gr_scannet_gt_path

DEFAULT_SCENE_LIST = [
    "scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00", "scene0140_00",
    "scene0200_00", "scene0347_00", "scene0400_00", "scene0590_00", "scene0645_00",
]

def run_scene_eval(
    scan_name,
    data_root,
    iteration,
    k,
    threshold,
    gt_json_name,
    save_vis,
    verbose,
):
    model_path = os.path.join(data_root, scan_name)
    gt_json_path = os.path.join(model_path, gt_json_name)
    ply_path = os.path.join(model_path, f"point_cloud/iteration_{iteration}/point_cloud.ply")

    if not os.path.exists(gt_json_path):
        raise FileNotFoundError(f"GT file not found: {gt_json_path}")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Point cloud not found: {ply_path}")

    data, objects = load_gt_objects(gt_json_path)
    if not objects:
        raise ValueError(f"No valid objects in GT file: {gt_json_path}")

    labels_file = data.get("file") or f"{scan_name}_vh_clean_2.labels.ply"
    labels_ply_path = os.path.join(model_path, labels_file)
    if not os.path.exists(labels_ply_path):
        raise FileNotFoundError(f"Labels file not found: {labels_ply_path}")

    ply_data = PlyData.read(ply_path)
    vertex_data = ply_data["vertex"].data
    model_points = np.vstack([vertex_data["x"], vertex_data["y"], vertex_data["z"]]).T
    num_points = len(model_points)

    scene_intersections = []
    scene_unions = []
    scene_corrects = []
    scene_totals = []
    scene_ious = []
    vis_dir_printed = False
    vis_subdir = "eval"

    for obj in objects:
        gt_id = obj.get("gt_id", [])
        label_list = obj.get("label", [])
        if not label_list:
            if verbose:
                print("  skip: empty label")
            continue
        valid_gt_ids = [x for x in gt_id if x is not None and int(x) >= 0]
        referring_expr = label_list[0] if isinstance(label_list[0], str) else str(label_list[0])

        if not valid_gt_ids:
            gt_mask = torch.zeros(num_points, dtype=torch.int64).cuda()
        else:
            gt_mask = build_gt_mask_from_gt_id(valid_gt_ids, labels_ply_path, num_points)
        gt_cnt = torch.sum(gt_mask).item()

        pred_path = os.path.join(
            model_path, "knn_label",
            f"{referring_expr}_fixed_labels_k{k}_t{threshold}.pth"
        )

        if not os.path.exists(pred_path):
            pred_cnt = 0
            if gt_cnt == 0:
                inter, uni, corr, tot = 1, 1, 1, 1
                cls_iou = 1.0
                print("  GT point count is 0; IoU set to 1")
            else:
                inter, uni, corr, tot = 0, max(gt_cnt, 1), 0, gt_cnt
                cls_iou = 0.0
                print("  GT point count > 0; IoU set to 0")
            if save_vis:
                if not vis_dir_printed:
                    vis_root = os.path.join(model_path, "semantic_slices", vis_subdir)
                    print(f"  [vis] scene output dir: {os.path.abspath(vis_root)}")
                    vis_dir_printed = True
                pred_mask_none = np.zeros(num_points, dtype=bool)
                save_semantic_slices(
                    model_points, pred_mask_none, referring_expr, model_path,
                    subdir=vis_subdir,
                )
        else:
            pred_data = torch.load(pred_path, map_location="cuda")
            pred_labels = pred_data["refined_label"]
            pred_mask = (pred_labels == 1)
            if pred_mask.dim() > 1:
                pred_mask = pred_mask.squeeze()
            if len(pred_mask) != num_points:
                raise ValueError(
                    f"Prediction count ({len(pred_mask)}) != point cloud count ({num_points}), "
                    f"file: {pred_path}, object: {referring_expr[:50]}..."
                )
            pred_mask = pred_mask.to(gt_mask.device)
            inter, uni, corr, tot = calculate_metrics_per_class(gt_mask, pred_mask)
            pred_cnt = pred_mask.sum().item()
            if gt_cnt == 0:
                if pred_cnt == 0:
                    cls_iou = 1.0
                    inter, uni = 1, 1
                else:
                    cls_iou = 0.0
            else:
                cls_iou = inter / uni if uni > 0 else 0.0

            if save_vis:
                if not vis_dir_printed:
                    vis_root = os.path.join(model_path, "semantic_slices", vis_subdir)
                    print(f"  [vis] scene output dir: {os.path.abspath(vis_root)}")
                    vis_dir_printed = True
                save_semantic_slices(
                    model_points, pred_mask, referring_expr, model_path,
                    subdir=vis_subdir,
                )

        scene_intersections.append(inter)
        scene_unions.append(uni)
        scene_corrects.append(corr)
        scene_totals.append(tot)
        scene_ious.append(cls_iou)

        if verbose:
            short_label = referring_expr[:50] + "..." if len(referring_expr) > 50 else referring_expr
            print(f"  {short_label:<55} | GT:{gt_cnt:<6} | Pred:{pred_cnt:<6} | IoU:{cls_iou:.4f}")

    mean_iou, accuracy, mean_acc = aggregate_scene_metrics(
        scene_intersections, scene_unions, scene_corrects, scene_totals
    )
    return mean_iou, accuracy, mean_acc, scene_ious

def main():
    parser = argparse.ArgumentParser(description="GR-ScanNet 3D referring segmentation evaluation")
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="GR-ScanNet data root",
    )
    parser.add_argument(
        "--scene_name", type=str, default=None,
        help="Evaluate a single scene (e.g. scene0400_00); overrides --scene_list",
    )
    parser.add_argument(
        "--scene_list", type=str, nargs="+", default=None,
        help="Scene list; default is 10 GR-ScanNet scenes",
    )
    parser.add_argument("--iteration", type=int, default=90000)
    parser.add_argument("--k", type=int, default=40, help="KNN k (must match render)")
    parser.add_argument("--threshold", type=float, default=0.8, help="KNN threshold (must match render)")
    parser.add_argument(
        "--gt_json", type=str, default=None,
        help="Relative GT JSON path; auto-detected if omitted",
    )
    parser.add_argument("--save_vis", type=bool, default=False, help="Save visualizations")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose logging")

    args = parser.parse_args()
    if getattr(args, "scene_name", None):
        scene_list = [args.scene_name.strip()]
    else:
        scene_list = args.scene_list or DEFAULT_SCENE_LIST
    print(f"Evaluating {len(scene_list)} scene(s): {scene_list}")

    mious, accs, maccs = [], [], []

    for scan_name in tqdm(scene_list, desc="  eval", leave=False):
        if args.gt_json:
            gt_json_name = args.gt_json
        else:
            text_dir = os.path.join(args.data_root, scan_name, f"{scan_name}_text")
            resolved = resolve_gr_scannet_gt_path(text_dir, f"{scan_name}.json")
            gt_json_name = (
                os.path.relpath(str(resolved), os.path.join(args.data_root, scan_name))
                if resolved is not None
                else os.path.join(f"{scan_name}_text", f"{scan_name}.json")
            )
        try:
            miou, acc, macc, _ = run_scene_eval(
                scan_name=scan_name,
                data_root=args.data_root,
                iteration=args.iteration,
                k=args.k,
                threshold=args.threshold,
                gt_json_name=gt_json_name,
                save_vis=args.save_vis,
                verbose=args.verbose,
            )
            mious.append(miou)
            accs.append(acc)
            maccs.append(macc)
            if args.verbose:
                print(f"    {scan_name} | mIoU: {miou:.4f} | Acc: {acc:.4f} | mAcc: {macc:.4f}")
        except Exception as e:
            print(f"    {scan_name} evaluation failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 70)
    print("gr_scannet evaluation summary")
    print("=" * 70)
    if mious:
        print(f"  mIoU: {np.nanmean(mious):.4f} | Acc: {np.nanmean(accs):.4f} | mAcc: {np.nanmean(maccs):.4f}")
    else:
        print("  (no data)")
    print("=" * 70)

if __name__ == "__main__":
    main()
