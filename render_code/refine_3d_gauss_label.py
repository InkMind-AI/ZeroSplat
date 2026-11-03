import torch
from tqdm import tqdm
import sys

def filter_by_backproject(unique_label, gaussians, view_proj_dict, views, back_ratio_threshold=0.8):

    fg_mask = (unique_label == 1)
    N_gaussians = unique_label.shape[0]
    N_fg = fg_mask.sum().item()
    print(f"\nBack-projection check: {N_fg} foreground Gaussians, {len(view_proj_dict)} views")
    if N_fg == 0:
        print("warning: no foreground Gaussians to verify; returning original labels")
        return unique_label

    bg_view_count = torch.zeros(N_gaussians, dtype=torch.long, device="cuda")
    visible_view_count = torch.zeros(N_gaussians, dtype=torch.float32, device="cuda")
    fg_indices = torch.where(fg_mask)[0]

    for view in tqdm(views, desc="Back-project foreground Gaussians", mininterval=1.0):
        view_name = view.image_name
        if view_name not in view_proj_dict:
            print(f"warning: no projection for view {view_name}, skipping")
            continue

        image_xys = view_proj_dict[view_name]  # [2, N_gaussians]
        fg_proj_x = image_xys[0, fg_mask]
        fg_proj_y = image_xys[1, fg_mask]
        fg_proj_2d = torch.stack([fg_proj_x, fg_proj_y], dim=1)  # [N_fg, 2]

        target_mask = view.objects.squeeze()  # [H, W]
        target_mask = (target_mask > 0).long()

        H, W = target_mask.shape[0], target_mask.shape[1]

        pixel_coords = fg_proj_2d.long()
        valid_x = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < W)
        valid_y = (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < H)
        is_in_screen = valid_x & valid_y
        valid_indices_local = torch.where(is_in_screen)[0]
        if len(valid_indices_local) == 0:
            continue
        valid_coords = pixel_coords[valid_indices_local]
        sampled_mask = target_mask[valid_coords[:, 1], valid_coords[:, 0]]
        is_bg_local = (sampled_mask == 0)

        global_valid_indices = fg_indices[valid_indices_local]

        visible_view_count[global_valid_indices] += 1
        bg_view_count[global_valid_indices[is_bg_local]] += 1

    bg_ratio = bg_view_count.float() / (visible_view_count + 1e-6)
    pseudo_fg_mask = (bg_ratio > back_ratio_threshold) & fg_mask & (visible_view_count > 0)
    N_pseudo = pseudo_fg_mask.sum().item()

    refined_label = unique_label.clone()
    refined_label[pseudo_fg_mask] = 0

    print("\nBack-projection filter stats:")
    print(f"background ratio threshold: {back_ratio_threshold}")
    print(f"original foreground Gaussians: {N_fg}")
    print(f"filtered artifact Gaussians: {N_pseudo}")
    print(f"foreground Gaussians after filter: {N_fg - N_pseudo}")
    return refined_label
