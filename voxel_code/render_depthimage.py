import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
from os import makedirs
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from scene import Scene
from gaussian_renderer import zero_render
from gaussian_renderer import GaussianModel
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
def render_viewpoints(model_path, iteration, views, gaussians, pipeline, background, skip_list):
    depth_path = os.path.join(model_path, "depth_image")
    makedirs(depth_path, exist_ok=True)
    print(f"Exporting depth maps to: {depth_path}")
    
    for view in tqdm(views, desc="Rendering Depth", mininterval=1.0):
        if view.image_name in skip_list:
            continue
        with torch.inference_mode():
            render_pkg = zero_render(view, gaussians, pipeline, background)
        
        depth_tensor = render_pkg["depth"] 
        depth_map = depth_tensor.detach().cpu().numpy()[0]
        depth_mm = (depth_map*1000).astype(np.uint16)
        save_name = os.path.join(depth_path, f"{view.image_name}.png")
        makedirs(os.path.dirname(save_name), exist_ok=True)
        cv2.imwrite(save_name, depth_mm)