"""Remove stale pipeline outputs before a fresh run."""

import os
import shutil

def _remove_path(path: str) -> bool:
    if not os.path.lexists(path):
        return False
    if os.path.islink(path) or os.path.isfile(path):
        os.remove(path)
    else:
        shutil.rmtree(path)
    return True

def clean_vlm_artifacts(scene_dir: str, vis_save_root: str | None = None) -> None:
    vis_root = vis_save_root or os.path.join(scene_dir, "vlm_visual_results")
    for path in (
        vis_root,
        os.path.join(scene_dir, "llm_ayal.json"),
        os.path.join(scene_dir, "llm_ayal_bbox.json"),
    ):
        if _remove_path(path):
            print(f"[clean] removed {path}")

def clean_sam3_artifacts(scene_dir: str) -> None:
    path = os.path.join(scene_dir, "sam3_mask")
    if _remove_path(path):
        print(f"[clean] removed {path}")

def clean_projection_artifacts(vis_save_root: str) -> None:
    path = os.path.join(vis_save_root, "proj.pth")
    if _remove_path(path):
        print(f"[clean] removed {path}")

def clean_render_artifacts(
    scene_dir: str,
    lift_out: str = "zero_out",
    knn_label: bool = True,
) -> None:
    names = [lift_out]
    if knn_label:
        names.append("knn_label")
    for name in names:
        path = os.path.join(scene_dir, name)
        if _remove_path(path):
            print(f"[clean] removed {path}")

def clean_mask_stage(
    scene_dir: str,
    vis_save_root: str,
    *,
    vlm_only: bool = False,
    skip_vlm: bool = False,
    proj_only: bool = False,
) -> None:
    if proj_only:
        clean_projection_artifacts(vis_save_root)
        return
    if vlm_only:
        clean_vlm_artifacts(scene_dir, vis_save_root)
        return
    if skip_vlm:
        clean_sam3_artifacts(scene_dir)
        clean_projection_artifacts(vis_save_root)
        return
    clean_vlm_artifacts(scene_dir, vis_save_root)
    clean_sam3_artifacts(scene_dir)
    clean_projection_artifacts(vis_save_root)
