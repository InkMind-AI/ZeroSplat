#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
import sys
from scene import Scene
import os
from os import path as osp
from tqdm import tqdm
from os import makedirs
# import imageio 
import cv2 
from render_code.refine_3d_gauss_label import filter_by_backproject
from render_code.filter_by_box_referencetext import filter_by_json_box
from gaussian_renderer import render
from gaussian_renderer import zero_render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from PIL import Image
import numpy as np
import colorsys
import torch
import torch.nn.functional as F
import random
from preprocess.pre_dataset import dataset_dir_name, get_prompt_render, select_test_image
from render_code.point_cloud_spread import knn_spread_main
from utils.pose_check import PoseAlignmentChecker

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, 
                target_name="",enable_conflict_filter = True,dataset_name = "",dataset_type = "",
               conflict_threshold=80,proj_file_path="",json_path=""):

    train_base_path = os.path.join(model_path, "zero_out", target_name, "train", "ours_{}".format(iteration))
    current_base_path = os.path.join(model_path, "zero_out", target_name, name, "ours_{}".format(iteration))
    render_path = os.path.join(current_base_path, "renders")
    gts_path = os.path.join(current_base_path, "gt")
    obj_mask_path = os.path.join(current_base_path, "pred_mask")
    depth_path = os.path.join(current_base_path, "depth_image")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    if name == "test":
        os.makedirs(obj_mask_path, exist_ok=True) 

    views_used = views

    all_obj_labels = None
    unique_label = None
    all_counts = None

    if name == "train":
        stats_counts_path = os.path.join(train_base_path, "used_count")
        makedirs(stats_counts_path, exist_ok=True)

        cur_count_dir = os.path.join(stats_counts_path, "name_{}.pth".format(target_name))
        proj_dict_dir = os.path.join(stats_counts_path, "name_{}_proj_dict.pth".format(target_name))
        view_proj_dict = {}

        if os.path.exists(cur_count_dir) and os.path.exists(proj_dict_dir):
            print(f"Found saved contribution matrix: {cur_count_dir}")
            print(f"Found saved projection coord dict: {proj_dict_dir}")
            all_counts = torch.load(cur_count_dir, map_location="cuda")
            view_proj_dict = torch.load(proj_dict_dir, map_location="cuda")
        else:
            print(f"[{target_name}] Computing contribution matrix (Train)...")
            pose_checker = PoseAlignmentChecker(tag=target_name)
            for idx, view in enumerate(tqdm(views_used, desc="Rendering progress", mininterval=1.0)):
                gt_mask = view.objects.to(torch.float32) / 255.
                render_pkg = zero_render(view, gaussians, pipeline, background, gt_mask=gt_mask)
                rendering = render_pkg["render"]
                class_weights = render_pkg["class_weights"]
                image_xys = render_pkg["image_xys"]

                view_proj_dict[view.image_name] = image_xys
                if all_counts is None:
                    all_counts = torch.zeros_like(class_weights)
                all_counts += class_weights
                gt = view.original_image[0:3, :, :]
                pose_checker.update(rendering, gt)
                os.makedirs(render_path, exist_ok=True)
                os.makedirs(gts_path, exist_ok=True)
                os.makedirs(depth_path, exist_ok=True)
                torchvision.utils.save_image(rendering, os.path.join(render_path, '{}'.format(view.image_name) + ".png"))
                torchvision.utils.save_image(gt, os.path.join(gts_path, '{}'.format(view.image_name) + ".png"))
                
            torch.save(all_counts, cur_count_dir)
            torch.save(view_proj_dict, proj_dict_dir)
            pose_checker.report()

            if all_counts is not None:
                all_counts = torch.nn.functional.normalize(all_counts, dim=0)
                unique_label = all_counts.max(dim=0)[1]
                conflict_save_path = os.path.join(
                    model_path, "zero_out", target_name, name,
                    "ours_{}".format(iteration), "label_refinement",
                )
                makedirs(conflict_save_path, exist_ok=True)

                if enable_conflict_filter and name == "train" and unique_label is not None:
                    unique_label = filter_by_backproject(
                        unique_label=unique_label,
                        gaussians=gaussians,
                        view_proj_dict=view_proj_dict,
                        views=views,
                        back_ratio_threshold=conflict_threshold / 100.0,
                    )
                    print(f"DEBUG: unique_label shape after backproject: {unique_label.shape}")
                    if dataset_type in ['Ref-lerf', 'gr_scannet', 'grlerf']:
                        unique_label = filter_by_json_box(
                            unique_label=unique_label,
                            views=views,
                            target_text=target_name,
                            back_ratio_threshold=80 / 100.0,
                            proj_file_path=proj_file_path,
                            json_path=json_path,
                        )
                    torch.save({
                        'refined_label': unique_label,
                        'vote_threshold': conflict_threshold / 100.0,
                    }, os.path.join(conflict_save_path, 'refined_labels.pth'))
                    print(f"  Saved refined labels: refined_labels.pth")

    refined_label_path = os.path.join(train_base_path, "label_refinement", 'refined_labels.pth')
    if os.path.exists(refined_label_path):
        if unique_label is None and all_obj_labels is None:
            print(f"[{name}] Loading trained labels: {refined_label_path}")
            refined_data = torch.load(refined_label_path, map_location="cuda")
            unique_label = refined_data['refined_label']
            print(f"Foreground Gaussians after loading: {(unique_label == 1).sum().item()}")
    else:
        if name == 'test':
            print(f"Error: trained label file not found: {refined_label_path}. Run Train first.")
            sys.exit(1)
        if unique_label is None and all_obj_labels is None:
            sys.exit(1)

    if name == "test":
        print(f"[{name}] Starting object rendering...")
        views_used = views
        conflict_save_path = os.path.join(
            model_path, "zero_out", target_name, "train",
            "ours_{}".format(iteration), "label_refinement",
        )
        refined_label_path = os.path.join(conflict_save_path, 'refined_labels.pth')
        if not os.path.exists(refined_label_path):
            print(f"Path does not exist: {refined_label_path}")
            sys.exit(1)
        if unique_label is None:
            print(f"Loading refined labels from train stage: {refined_label_path}")
            refined_data = torch.load(refined_label_path, map_location="cuda")
            unique_label = refined_data['refined_label']
            print(f"Foreground Gaussians after loading: {(unique_label == 1).sum().item()}")

        obj_used_mask = (unique_label == 1)
        # eval reads test/.../pred_mask/*.png
        obj_render_path = os.path.join(
            model_path, "zero_out", target_name, name,
            "ours_{}".format(iteration), "obj_{:03d}".format(1),
        )
        os.makedirs(obj_render_path, exist_ok=True)

        for idx, view in enumerate(tqdm(views_used, desc="Rendering object {:03d}".format(1), mininterval=1.0)):
            obj_pkg = zero_render(
                view, gaussians, pipeline, background, keep_color_mask=obj_used_mask
            )
            rendering = obj_pkg["render"]
            gt = view.original_image[0:3, :, :]
            mask = (rendering > (30.0 / 255.0)).any(dim=0).float()
            torchvision.utils.save_image(mask, os.path.join(obj_mask_path, '{}'.format(view.image_name) + ".png"))
            torchvision.utils.save_image(rendering, os.path.join(obj_render_path, '{}'.format(view.image_name) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{}'.format(view.image_name) + ".png"))
        

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, 
                  dataset_type,pts_path, label_path, save_dir, k, threshold, dataset_name, proj_file_path="",json_path="",if_knn_spread=True):
    if hasattr(dataset, 'target_name'):
        if isinstance(dataset.target_name, list):
            target_name_list = dataset.target_name
        elif isinstance(dataset.target_name, str):
            target_name_list = [dataset.target_name]
        else:
            target_name_list = ["default"]
    else:
        target_name_list = ["default"]

    gaussians = GaussianModel(dataset.sh_degree)
    if isinstance(iteration, int):
        ply_path = os.path.join(dataset.model_path, "point_cloud",
                                "iteration_" + str(iteration), "point_cloud.ply")
        if os.path.exists(ply_path):
            gaussians.load_ply(ply_path)
    
    for target_name in target_name_list:
        print(f"\n{'='*60}")
        print(f"Processing target: {target_name}")
        print(f"{'='*60}")
        mask_root = os.path.join(dataset.model_path, dataset.object_path, target_name)
        if not os.path.exists(mask_root):
            no_valid_mask = True
        else:
            mask_image_files = [f for f in os.listdir(mask_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            no_valid_mask = (len(mask_image_files) == 0)
        if no_valid_mask == True and dataset_type in ["scannet","gr_scannet"]:
            continue
        if no_valid_mask == True and dataset_type not in ["scannet","gr_scannet"]:
            ours_dir = os.path.join(dataset.model_path, "zero_out", target_name, "test", f"ours_{iteration}")
            out_dir = os.path.join(ours_dir, "obj_001")
            mask_dir = os.path.join(ours_dir, "pred_mask")
            os.makedirs(out_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            test_frame_names = getattr(dataset, "test_image_names", []) or []
            if dataset_type in ['grlerf'] and dataset_name in ['figurines']:
                default_h, default_w = 728, 986
            elif dataset_type in ['lerf'] and dataset_name in ['ramen']:
                default_h, default_w = 731, 988
            elif dataset_type in ['Ref-lerf'] and dataset_name in ['ramen']:
                default_h, default_w = 731, 988
            elif dataset_type in ['grlerf'] and dataset_name in ['waldo_kitchen']:
                default_h, default_w = 725, 985
            elif dataset_type in ['grlerf'] and dataset_name in ['ramen']:
                default_h, default_w = 731, 988
            else:
                default_h, default_w = 730, 988

            for frame_name in test_frame_names:
                black = np.zeros((default_h, default_w), dtype=np.uint8)
                Image.fromarray(black, mode="L").save(os.path.join(out_dir, frame_name + ".png"))
                Image.fromarray(black, mode="L").save(os.path.join(mask_dir, frame_name + ".png"))
            print(f"Target {target_name} has no valid masks; wrote {len(test_frame_names)} black test frames to {out_dir} and {mask_dir}, skipping")
            continue

        dataset.target_name = target_name
        
        with torch.inference_mode():
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
            bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, 
                             target_name=dataset.target_name,enable_conflict_filter = True,dataset_name = dataset_name,dataset_type = dataset_type,proj_file_path=proj_file_path,json_path=json_path)
            if if_knn_spread ==True:
                knn_spread_main(pts_path=pts_path, label_path=label_path, save_dir=save_dir, k=k, threshold=threshold, dataset_name=dataset_name, dataset_type=dataset_type,text=dataset.target_name)

            if dataset_type in ["Ref-lerf", "lerf","grlerf"]:
                if not skip_test:
                    render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, 
                             target_name=dataset.target_name,enable_conflict_filter = False)
        del scene
        del background
        torch.cuda.empty_cache()

        print(f"Finished target {target_name}.\n")

    del gaussians
    torch.cuda.empty_cache()

if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.abspath(__file__))

    parser = ArgumentParser(description="Gaussian labeling + Zero render")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    # --- dataset ---
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="grlerf",
        choices=["Ref-lerf", "scannet", "lerf", "gr_scannet", "grlerf"],
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        nargs="+",
        default=["ramen"],
        help="One or more scene folder names",
    )
    parser.add_argument("--base_data_dir", type=str, required=True)

    # --- paths (optional overrides; normally auto-derived per scene) ---
    # NOTE: model_path / source_path / object_path / target_name are already
    # registered by ModelParams; use set_defaults instead of add_argument.
    parser.add_argument("--proj_file_path", type=str, default="")
    parser.add_argument("--json_path", type=str, default="")

    # --- algorithm ---
    parser.add_argument("--k", type=int, default=40, help="KNN neighbor count")
    parser.add_argument("--threshold", type=float, default=0.8, help="KNN repair threshold")
    parser.add_argument("--if_knn_spread", type=bool, default=True)
    parser.add_argument("--obj_id", default=-1, type=int)
    parser.set_defaults(
        model_path="",
        source_path="",
        object_path="sam3_mask",
        target_name="",
    )

    # --- run ---
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--keep_artifacts",
        action="store_true",
        help="Keep previous zero_out / knn_label (default: clean before run)",
    )

    args = get_combined_args(parser)
    base_data_dir = os.path.abspath(args.base_data_dir)
    # sentinel=True + no cfg_args: ModelParams attrs are missing; fill defaults
    _defaults = {
        "sh_degree": 3,
        "white_background": False,
        "data_device": "cuda",
        "eval": True,
        "resolution": -1,
        "model_path": "",
        "source_path": "",
        "object_path": "sam3_mask",
        "target_name": "",
        "test_image_names": [],
    }
    for _k, _v in _defaults.items():
        if getattr(args, _k, None) is None:
            setattr(args, _k, _v)

    if args.dataset_type in ["Ref-lerf", "lerf", "grlerf"]:
        args.images = "images"
        args.iteration = 30000
        iter_name = 30000
    elif args.dataset_type in ["scannet", "gr_scannet"]:
        args.images = "color"
        args.iteration = 90000
        iter_name = 90000
    else:
        print("Invalid dataset_type")
        sys.exit(1)

    safe_state(args.quiet)

    print(f"\n{'=' * 80}")
    print(f"Batch render: {len(args.dataset_name)} scene(s)")
    print(f"dataset_type: {args.dataset_type}")
    print(f"scenes: {args.dataset_name}")
    print(f"{'=' * 80}\n")

    for scene_idx, current_scene in enumerate(args.dataset_name, 1):
        print(f"\n{'#' * 80}")
        print(f"Scene [{scene_idx}/{len(args.dataset_name)}]: {current_scene}")
        print(f"{'#' * 80}\n")

        scene_dir = os.path.join(
            base_data_dir, dataset_dir_name(args.dataset_type), current_scene
        )
        args.model_path = scene_dir
        args.source_path = scene_dir
        args.pts_path = os.path.join(
            scene_dir, f"point_cloud/iteration_{iter_name}/point_cloud.ply"
        )
        args.proj_file_path = os.path.join(scene_dir, "vlm_visual_results", "proj.pth")
        args.json_path = os.path.join(
            scene_dir, "vlm_visual_results", "localization_results.json"
        )
        args.label_path = os.path.join(args.model_path, "zero_out")
        args.save_dir = os.path.join(args.model_path, "knn_label")

        args.test_image_names = select_test_image(args.dataset_type, current_scene)
        args.target_name = get_prompt_render(
            args.dataset_type, current_scene, base_data_dir
        )

        print(f"scene_dir: {args.model_path}")
        print(f"test images: {args.test_image_names}")
        print(f"targets: {args.target_name}")
        print(f"Rendering {args.model_path}\n")

        if not getattr(args, "keep_artifacts", False):
            from utils.run_artifacts import clean_render_artifacts

            clean_render_artifacts(scene_dir, lift_out="zero_out")

        try:
            render_sets(
                model.extract(args),
                args.iteration,
                pipeline.extract(args),
                args.skip_train,
                args.skip_test,
                args.dataset_type,
                args.pts_path,
                args.label_path,
                args.save_dir,
                args.k,
                args.threshold,
                current_scene,
                proj_file_path=args.proj_file_path,
                json_path=args.json_path,
                if_knn_spread=args.if_knn_spread,
            )
            print(f"\n[√] Scene {current_scene} done.")
        except Exception as e:
            print(f"\n[x] Scene {current_scene} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'=' * 80}")
    print("All scenes finished.")
    print(f"{'=' * 80}\n")
