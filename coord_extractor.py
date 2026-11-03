import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
from os import makedirs
from scene import Scene
from gaussian_renderer import zero_render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import json
from preprocess.pre_dataset import select_test_image

def get_proxy(all_views, target_image_names, gaussians, pipeline, background):
    view_proj_dict = {}

    view_map = {}
    for v in all_views:
        base_name = os.path.basename(v.image_name) # frame_00032.jpg -> frame_00032
        root_name = os.path.splitext(base_name)[0]
        view_map[root_name] = v

    for name in tqdm(target_image_names, desc="Calculating Projections", mininterval=1.0):
        

        search_key = os.path.splitext(os.path.basename(name))[0]
        
        if search_key not in view_map:

            if name in view_map:
                view = view_map[name]
            else:
                print(f"warning: view not found in scene {name} (or {search_key}), skipping")
                continue
        else:
            view = view_map[search_key]

        with torch.inference_mode():
            render_pkg = zero_render(view, gaussians, pipeline, background)
        proj_xy_pixel = render_pkg["image_xys"]
        proj_xy_pixel = proj_xy_pixel.detach()
        view_proj_dict[view.image_name] = proj_xy_pixel

    return view_proj_dict

def get_all_unique_images(json_path):
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if not data:
        print("JSON file is empty")
        return []

    unique_images_set = set()

    for description, inner_dict in data.items():
      
        for image_name_with_ext in inner_dict.keys():
    
            name_no_ext = os.path.splitext(image_name_with_ext)[0]
          
            unique_images_set.add(name_no_ext)
    final_image_names = sorted(list(unique_images_set))
        
    print(f"Extracted {len(final_image_names)} unique image(s)")
    return final_image_names

def get_pro_xy_main(dataset_type, dataset_name, model_path, source_path, 
                    image_path, proj_dict_dir, iteration=30000, quiet=False):
    parser = ArgumentParser(description="Depth Export Script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=iteration, type=int)
    parser.add_argument("--quiet", action="store_true", default=quiet)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--dataset_name", type=str, default=dataset_name)
    parser.add_argument("--dataset_type", type=str, default=dataset_type)
    parser.add_argument("--image_path", default=image_path)
    parser.add_argument("--proj_dict_dir", default=proj_dict_dir)
    parser.set_defaults(
        target_name="",
        object_path="images",
        model_path=model_path,
        source_path=source_path,
    )

    # Called from mask_generator / main with programmatic kwargs; do not re-parse
    # parent CLI (e.g. --api_key / --base_url / --base_data_dir).
    _argv = sys.argv
    try:
        sys.argv = [_argv[0]]
        args = get_combined_args(parser)
    finally:
        sys.argv = _argv

    _defaults = {
        "sh_degree": 3,
        "white_background": False,
        "data_device": "cuda",
        "eval": True,
        "resolution": -1,
    }
    for key, val in _defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)
    args.images = "images"
    args.test_image_names = select_test_image(args.dataset_type, args.dataset_name)
    safe_state(args.quiet)

    with torch.inference_mode():

        gaussians = GaussianModel(args.sh_degree)
        scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
        
        bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        all_views = scene.getTrainCameras()
        target_image_names = get_all_unique_images(args.image_path)

        proj = get_proxy(
            all_views=all_views,
            target_image_names = target_image_names,
            gaussians=gaussians,
            pipeline=pipeline,
            background=background,
        )
        proj_dict_dir = args.proj_dict_dir
        torch.save(proj, proj_dict_dir)

   
