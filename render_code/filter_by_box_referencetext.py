import torch
from tqdm import tqdm
import sys
import json
import os

def filter_by_json_box(unique_label, views, target_text, back_ratio_threshold=0.8, proj_file_path="", json_path="", min_views_to_filter=8):
    if os.path.exists(proj_file_path):
        view_proj_dict = torch.load(proj_file_path, map_location="cuda")
        print(f"Loaded projections for {len(view_proj_dict)} views")
    else:
        print(f"File not found: {proj_file_path}")
        view_proj_dict = {}

    if not os.path.exists(json_path):
        print(f"localization_results.json not found: {json_path}")
        return unique_label

    with open(json_path, 'r', encoding='utf-8') as f:
        localization_results = json.load(f)

    if target_text not in localization_results:
        print(f"Text '{target_text}' not in JSON; skipping filter")
        return unique_label

    target_images_dict = localization_results[target_text]
    target_image_names = list(target_images_dict.keys())

    fg_mask = (unique_label == 1)
    N_gaussians = unique_label.shape[0]
    N_fg = fg_mask.sum().item()

    print(f"\n[BBox filter] target: {target_text}")
    print(f"views: {target_image_names}")
    print(f"foreground Gaussians to verify: {N_fg}")

    if N_fg == 0:
        print("warning: no foreground Gaussians to verify; returning original labels")
        return unique_label

    bg_view_count = torch.zeros(N_gaussians, dtype=torch.long, device="cuda")
    visible_view_count = torch.zeros(N_gaussians, dtype=torch.float32, device="cuda")

    fg_indices = torch.where(fg_mask)[0]

    if len(views) > 0:
        H, W = views[0].image_height, views[0].image_width

    for view_name in tqdm(target_image_names, desc=f"BBox filter: {target_text}"):
        search_key = os.path.splitext(view_name)[0]
        if search_key not in view_proj_dict:
            print(f"warning: no projection for view {search_key}, skipping")
            continue

        box_info = target_images_dict[view_name]
        if "box" not in box_info:
            continue
        raw_box = box_info["box"]
        if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4 and all(isinstance(v, (int, float)) for v in raw_box):
            xmin, ymin, xmax, ymax = raw_box
        elif isinstance(raw_box, (list, tuple)) and len(raw_box) > 0:
            boxes = []
            for b in raw_box:
                if isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
                    boxes.append(b)
            if not boxes:
                continue
            xmin = min(b[0] for b in boxes)
            ymin = min(b[1] for b in boxes)
            xmax = max(b[2] for b in boxes)
            ymax = max(b[3] for b in boxes)
        else:
            continue

        image_xys = view_proj_dict[search_key]  # [2, N_gaussians]

        fg_proj_x = image_xys[0, fg_mask]
        fg_proj_y = image_xys[1, fg_mask]

        x_int = fg_proj_x.long()
        y_int = fg_proj_y.long()

        valid_x = (x_int >= 0) & (x_int < W)
        valid_y = (y_int >= 0) & (y_int < H)
        is_in_screen = valid_x & valid_y

        valid_indices_local = torch.where(is_in_screen)[0]
        if len(valid_indices_local) == 0:
            continue

        global_valid_indices = fg_indices[valid_indices_local]
        visible_view_count[global_valid_indices] += 1

        x_screen = fg_proj_x[valid_indices_local]
        y_screen = fg_proj_y[valid_indices_local]

        in_box = (x_screen >= xmin) & (x_screen <= xmax) & \
                 (y_screen >= ymin) & (y_screen <= ymax)

        is_outside_box = ~in_box

        bg_view_count[global_valid_indices[is_outside_box]] += 1

    bg_ratio = bg_view_count.float() / (visible_view_count + 1e-6)

    in_enough_views = (visible_view_count >= min_views_to_filter)
    pseudo_fg_mask = (bg_ratio > back_ratio_threshold) & fg_mask & in_enough_views
    N_pseudo = pseudo_fg_mask.sum().item()
    N_skipped = ((bg_ratio > back_ratio_threshold) & fg_mask & (visible_view_count > 0) & ~in_enough_views).sum().item()

    refined_label = unique_label.clone()
    refined_label[pseudo_fg_mask] = 0

    return refined_label
