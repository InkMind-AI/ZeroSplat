#!/usr/bin/env python3

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from tqdm import tqdm

from lerf_reflerf_common import (
    LERF_SCENES,
    load_image_as_binary,
    calculate_iou,
    aggregate_metrics,
    get_frame_objects,
)
from lerf_reflerf_vis import save_mask_vis

def run_scene_eval(
    scene_name,
    gt_base,
    pred_base,
    iteration=30000,
    save_vis=False,
    vis_dir=None,
    verbose=True,
):

    if scene_name not in LERF_SCENES:
        raise ValueError(f"Unknown scene: {scene_name}; supported: {list(LERF_SCENES.keys())}")

    frame_names = LERF_SCENES[scene_name]

    frame_dict, valid_frames = get_frame_objects(gt_base, frame_names)

    if len(valid_frames) != len(frame_names):
        print(f"warning: scene {scene_name} missing some test frames")
        print(f"  expected: {len(frame_names)}, found: {len(valid_frames)}")

    if verbose:
        print(f"\nScene {scene_name} test frames:")
        for frame, objs in frame_dict.items():
            print(f"  {frame}: {len(objs)} object(s)")

    ious = []
    skipped_count = 0

    for frame_name, obj_list in frame_dict.items():
        if verbose:
            print(f"\nEvaluating frame: {frame_name}")

        for obj_name in obj_list:
            if verbose:
                print(f"  object: {obj_name}")

            gt_obj_path = os.path.join(gt_base, frame_name, f"{obj_name}.jpg")

            if not os.path.exists(gt_obj_path):
                print(f"    GT missing: {gt_obj_path}, skipping")
                skipped_count += 1
                continue

            pred_obj_path = os.path.join(
                pred_base,
                obj_name,
                "test",
                f"ours_{iteration}",
                "pred_mask",
                f"{frame_name}.png"
            )

            if not os.path.exists(pred_obj_path):
                if verbose:
                    print(f"    prediction missing: {pred_obj_path}, skipping")
                skipped_count += 1
                continue

            try:
                mask_gt = load_image_as_binary(gt_obj_path, is_png=False)
                mask_pred = load_image_as_binary(pred_obj_path, is_png=True)

                if mask_gt.shape != mask_pred.shape:
                    print(f"    mask size mismatch: GT {mask_gt.shape} vs Pred {mask_pred.shape}, skipping")
                    skipped_count += 1
                    continue

                iou = calculate_iou(mask_gt, mask_pred)
                ious.append(iou)

                if verbose:
                    print(f"    IoU: {iou:.4f}")

                if save_vis and vis_dir is not None:
                    save_mask_vis(mask_gt, mask_pred, frame_name, obj_name, vis_dir)

            except Exception as e:
                print(f"    IoU computation failed: {e}, skipping")
                skipped_count += 1
                continue

    mean_iou, acc_025, acc_050 = aggregate_metrics(ious)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Scene {scene_name} results:")
        print(f"  samples: {len(ious)}")
        print(f"  skipped: {skipped_count}")
        print(f"  mIoU: {mean_iou:.4f}")
        print(f"  Acc@0.25: {acc_025:.4f}")
        print(f"  Acc@0.5: {acc_050:.4f}")
        print(f"{'='*60}")

    return mean_iou, acc_025, acc_050, ious

def main():
    parser = argparse.ArgumentParser(description="LERF open-vocabulary segmentation evaluation")
    parser.add_argument("--scene_name", type=str, default="waldo_kitchen",
                        choices=list(LERF_SCENES.keys()),
                        help="Scene name")
    parser.add_argument("--gt_base", type=str, required=True, help="GT mask root")
    parser.add_argument("--pred_base", type=str, required=True, help="Prediction mask root")
    parser.add_argument("--iteration", type=int, default=30000,
                        help="Training iteration")
    parser.add_argument("--save_vis", action="store_true",
                        help="Save visualizations")
    parser.add_argument("--vis_dir", type=str, default=None,
                        help="Visualization output directory")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Verbose logging")

    args = parser.parse_args()

    if args.save_vis and args.vis_dir is None:
        args.vis_dir = os.path.join(args.pred_base, f"eval_vis_{args.scene_name}")

    try:
        mean_iou, acc_025, acc_050, _ = run_scene_eval(
            scene_name=args.scene_name,
            gt_base=args.gt_base,
            pred_base=args.pred_base,
            iteration=args.iteration,
            save_vis=args.save_vis,
            vis_dir=args.vis_dir,
            verbose=args.verbose,
        )

        print(f"\n{'='*60}")
        print(f"LERF evaluation done — scene: {args.scene_name}")
        print(f"{'='*60}")
        print(f"  mIoU:      {mean_iou:.4f}")
        print(f"  Acc@0.25:  {acc_025:.4f}")
        print(f"  Acc@0.5:   {acc_050:.4f}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
