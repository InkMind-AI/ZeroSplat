"""Step A: export depth maps + WGSC keyframe sampling."""
import os

import torch
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from preprocess.pre_dataset import dataset_dir_name, select_test_image
from scene import Scene
from utils.general_utils import safe_state
from voxel_code.render_depthimage import render_viewpoints
from voxel_code.voxel import voxel_grid_main

ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = ArgumentParser(description="Depth export + WGSC keyframe sampling")
    ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument(
        "--dataset_type",
        type=str,
        default="grlerf",
        choices=["grlerf", "gr-lerf", "Ref-lerf", "gr_scannet"],
        help="Dataset family",
    )
    parser.add_argument("--dataset_name", type=str, default="ramen", help="Scene folder name")
    parser.add_argument(
        "--base_data_dir",
        type=str,
        required=True,
        help="Root containing <dataset_type>/<dataset_name> (alias of data root)",
    )
    parser.add_argument("--data_root", type=str, default="", help="Alias of --base_data_dir")
    parser.add_argument("--video_dir", type=str, default="")
    parser.add_argument("--base_dir", type=str, default=None, help="Scene dir override")
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--scene_name",
        type=str,
        default="",
        help="Alias of --dataset_name (scene folder)",
    )

    args = get_combined_args(parser)
    # sentinel=True leaves ModelParams defaults as None when no cfg_args exists
    _defaults = {
        "sh_degree": 3,
        "white_background": False,
        "data_device": "cuda",
        "eval": True,
        "resolution": -1,
        "images": "images",
        "object_path": "images",
        "target_name": "",
        "test_image_names": [],
    }
    for key, val in _defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)

    dataset_type = args.dataset_type
    scene_name = (getattr(args, "scene_name", None) or args.dataset_name)
    base_data_dir = os.path.abspath(getattr(args, "data_root", None) or args.base_data_dir)
    video_dir = getattr(args, "video_dir", None) or os.path.join(
        base_data_dir, dataset_dir_name(dataset_type)
    )
    base_dir = getattr(args, "base_dir", None) or os.path.join(video_dir, scene_name)

    args.model_path = base_dir
    args.source_path = base_dir
    args.target_name = ""

    if dataset_type == "gr_scannet":
        args.images = "color"
        images_dir = os.path.join(base_dir, "color")
    else:
        args.images = "images"
        images_dir = os.path.join(base_dir, "images")

    safe_state(args.quiet)

    test_image_names = select_test_image(dataset_type, scene_name)
    args.test_image_names = test_image_names

    # voxel.py helper namespace
    args.scene_name = scene_name
    args.dataset_name = dataset_type  # historical: family name
    args.dataset_type = dataset_type
    args.video_dir = video_dir
    args.images_dir = images_dir
    args.output_filename = os.path.join(base_dir, "scannet_weighted_frames.json")
    args.ply_path = os.path.join(
        base_dir, f"point_cloud/iteration_{args.iteration}/point_cloud.ply"
    )

    print(f"[*] dataset_type={dataset_type} scene={scene_name}")
    print(f"[*] test_image_names: {test_image_names}")
    print(f"[*] images_dir: {images_dir}")
    print(f"[*] output_filename: {args.output_filename}")
    print(f"[*] ply_path: {args.ply_path}")

    pipe = pipeline.extract(args)
    with torch.no_grad():
        gaussians = GaussianModel(args.sh_degree)
        scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
        bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        render_viewpoints(
            model_path=args.model_path,
            iteration=args.iteration,
            views=scene.getTrainCameras(),
            gaussians=gaussians,
            pipeline=pipe,
            background=background,
            skip_list=args.test_image_names,
        )

    print("\n[√] Depth export done.")
    voxel_grid_main(args)

if __name__ == "__main__":
    main()
