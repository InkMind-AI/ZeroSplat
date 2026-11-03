#!/usr/bin/env python3
import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_EVAL = os.path.dirname(_SCRIPT_DIR)   # eval/
_PROJECT_ROOT = os.path.dirname(_ROOT_EVAL)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _ROOT_EVAL not in sys.path:
    sys.path.insert(0, _ROOT_EVAL)

import numpy as np
from tqdm import tqdm

from eval.lerf_reflerf.lerf_reflerf_common import (
    REFLERF_SCENES,
    load_image_as_binary,
    calculate_iou,
    aggregate_metrics,
    get_frame_objects,
)
from eval.lerf_reflerf.lerf_reflerf_vis import save_mask_vis

def load_json_mapping(dataset_type,json_path, json_names):

    try:
        from preprocess.read_reftext import batch_process_json
        return batch_process_json(dataset_type,json_path, json_names)
    except ImportError:
        print("error: cannot import preprocess.read_reftext.batch_process_json")
        print("Ref-LERF evaluation requires JSON mapping from batch_process_json")
        sys.exit(1)
    except Exception as e:
        print(f"failed to load JSON mapping: {e}")
        sys.exit(1)

def build_frame_name_mapping(frame_dict, data_dict, verbose=False):
    final_frame_dict = {}

    for frame_name, obj_list in frame_dict.items():
        json_key = f"{frame_name}.json"
        json_mapping = data_dict.get(json_key, {})

        if not json_mapping:
            print(f"warning: no JSON mapping for frame {frame_name}, skipping")
            continue

        mapped_names = {}

        for obj_name in obj_list:
            if obj_name in json_mapping:
                sentences = json_mapping[obj_name]
                if sentences:

                    pred_folder_name = sentences[0]
                    mapped_names[obj_name] = pred_folder_name
                else:
                    print(f"warning: empty JSON value for object {obj_name} in frame {frame_name}")
            else:
                print(f"warning: object '{obj_name}' not found in JSON for frame {frame_name}")

        final_frame_dict[frame_name] = mapped_names

    if verbose:
        print("\nObject name mapping:")
        for frame, mapping in final_frame_dict.items():
            print(f"  {frame}:")
            for orig, pred in mapping.items():
                print(f"    {orig} -> {pred}")

    return final_frame_dict

def run_scene_eval(
    dataset_type,
    scene_name,
    gt_base,
    pred_base,
    json_path,
    iteration=30000,
    save_vis=False,
    vis_dir=None,
    verbose=True,
):

    if scene_name not in REFLERF_SCENES:
        raise ValueError(f"Unknown scene: {scene_name}; supported: {list(REFLERF_SCENES.keys())}")

    frame_names = REFLERF_SCENES[scene_name]

    frame_dict, valid_frames = get_frame_objects(gt_base, frame_names)

    if len(valid_frames) != len(frame_names):
        print(f"warning: scene {scene_name} missing some test frames")
        print(f"  expected: {len(frame_names)}, found: {len(valid_frames)}")

    json_names = [f"{frame}.json" for frame in valid_frames]
    data_dict = load_json_mapping(dataset_type,json_path, json_names)

    final_frame_dict = build_frame_name_mapping(frame_dict, data_dict, verbose=verbose)

    if verbose:
        print(f"\nScene {scene_name} test frames:")
        for frame, objs in frame_dict.items():
            print(f"  {frame}: {len(objs)} object(s)")

    ious = []
    skipped_count = 0

    for frame_name, obj_list in frame_dict.items():
        if verbose:
            print(f"\nEvaluating frame: {frame_name}")

        frame_mapping = final_frame_dict.get(frame_name, {})

        for obj_name in obj_list:
            if verbose:
                print(f"  object: {obj_name}")

            gt_obj_path = os.path.join(gt_base, frame_name, f"{obj_name}.png")

            if not os.path.exists(gt_obj_path):
                print(f" GT missing: {gt_obj_path}, skipping")
                skipped_count += 1
                continue

            pred_folder_name = frame_mapping.get(obj_name)
            if not pred_folder_name:
                print(f" no prediction folder mapping for '{obj_name}', skipping")
                skipped_count += 1
                continue

            pred_obj_path = os.path.join(
                pred_base,
                pred_folder_name,
                "test",
                f"ours_{iteration}",
                "pred_mask",
                f"{frame_name}.png"
            )

            if not os.path.exists(pred_obj_path):
                if verbose:
                    print(f" prediction missing: {pred_obj_path}, skipping")
                skipped_count += 1
                continue

            try:
                mask_gt = load_image_as_binary(gt_obj_path, is_png=True)
                mask_pred = load_image_as_binary(pred_obj_path, is_png=True)

                if mask_gt.shape != mask_pred.shape:
                    print(f" mask size mismatch: GT {mask_gt.shape} vs Pred {mask_pred.shape}, skipping")
                    skipped_count += 1
                    continue

                gt_all_black = (np.sum(mask_gt) == 0)
                if gt_all_black:
                    iou = 1.0 if np.sum(mask_pred) == 0 else 0.0
                else:
                    iou = calculate_iou(mask_gt, mask_pred)
                ious.append(iou)

                if verbose:
                    print(f" IoU: {iou:.4f}")

                if save_vis and vis_dir is not None:
                    save_mask_vis(mask_gt, mask_pred, frame_name, obj_name, vis_dir)

            except Exception as e:
                print(f" IoU computation failed: {e}, skipping")
                skipped_count += 1
                continue

    mean_iou, acc_025, acc_050 = aggregate_metrics(ious)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Scene {scene_name} results:")
        print(f"  samples: {len(ious)}")
        print(f"  skipped: {skipped_count}")
        print(f"  mIoU: {mean_iou:.7f}")
        print(f"  Acc@0.25: {acc_025:.4f}")
        print(f"  Acc@0.5: {acc_050:.4f}")
        print(f"{'='*60}")

    return mean_iou, acc_025, acc_050, ious

def main():
    parser = argparse.ArgumentParser(description="Ref-LERF semantic segmentation evaluation")
    parser.add_argument("--scene_name", type=str, default=os.environ.get("SCENE_NAME", "waldo_kitchen"),
                        choices=list(REFLERF_SCENES.keys()),
                        help="Scene name (or env SCENE_NAME)")
    parser.add_argument("--gt_base", type=str, required=True, help="GT mask root")
    parser.add_argument("--pred_base", type=str, required=True, help="Prediction mask root")
    parser.add_argument("--json_path", type=str, required=True, help="JSON root directory")
    parser.add_argument("--iteration", type=int, default=30000,
                        help="Training iteration")
    parser.add_argument("--save_vis", action="store_true",
                        help="Save visualizations")
    parser.add_argument("--vis_dir", type=str, default=None,
                        help="Visualization output directory")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Verbose logging")
    parser.add_argument("--dataset_type", type=str, default="grlerf",
                        help="Dataset type for JSON mapping")

    args = parser.parse_args()
    print(f"[eval] scene={args.scene_name}")
    print(f"[eval] pred_base={args.pred_base}")
    print(f"[eval] gt_base={args.gt_base}")
    print(f"[eval] json_path={args.json_path}")

    if args.save_vis and args.vis_dir is None:
        args.vis_dir = os.path.join(args.pred_base, f"eval_vis_{args.scene_name}")

    try:
        mean_iou, acc_025, acc_050, _ = run_scene_eval(
            dataset_type = args.dataset_type,
            scene_name=args.scene_name,
            gt_base=args.gt_base,
            pred_base=args.pred_base,
            json_path=args.json_path,
            iteration=args.iteration,
            save_vis=args.save_vis,
            vis_dir=args.vis_dir,
            verbose=args.verbose,
        )

        print(f"\n{'='*60}")
        print(f"Ref-LERF evaluation done — scene: {args.scene_name}")
        print(f"{'='*60}")
        print(f"  mIoU:      {mean_iou:.7f}")
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
