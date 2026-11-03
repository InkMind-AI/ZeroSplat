import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

from utils.required_cli import require_nonempty, require_vlm_endpoint

def _run(name, cmd, env=None):
    print(f"[Pipeline] {name}")
    print(f"[Pipeline] {' '.join(cmd)}")
    ret = subprocess.run(cmd, cwd=ROOT, env=env)
    if ret.returncode != 0:
        raise SystemExit(f"[Pipeline] {name} failed (exit={ret.returncode})")
    print(f"\n[Pipeline] {name} finished.\n")

def parse_args():
    p = argparse.ArgumentParser(
        description="ZeroSplat pipeline: voxel -> VLM API/SAM3 -> render"
    )

    # --- dataset ---
    p.add_argument(
        "--dataset_type",
        type=str,
        default="grlerf",
        choices=["grlerf", "Ref-lerf", "lerf", "gr_scannet", "scannet"],
    )
    p.add_argument("--dataset_name", type=str, default="ramen", help="Scene folder name")
    p.add_argument(
        "--base_data_dir",
        type=str,
        required=True,
        help="Root containing <dataset_type>/<dataset_name>",
    )
    p.add_argument("--iteration", type=int, default=None)

    # --- voxel ---
    p.add_argument("--voxel_size", type=float, default=0.1)

    # --- VLM API (OpenAI-compatible, e.g. DashScope or local vLLM) ---
    p.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key (required unless --proj_only or --skip_mask). "
        "Local vLLM: use EMPTY.",
    )
    p.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL",
    )
    p.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name served at base_url",
    )

    # --- python ---
    p.add_argument(
        "--gs_python",
        type=str,
        required=True,
        help="Python interpreter for pipeline stages",
    )

    # --- render ---
    p.add_argument("--k", type=int, default=40)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_test", action="store_true")
    p.add_argument("--quiet", action="store_true")

    # --- stage switches ---
    p.add_argument("--skip_voxel", action="store_true")
    p.add_argument("--skip_mask", action="store_true")
    p.add_argument("--skip_render", action="store_true")
    p.add_argument(
        "--keep_artifacts",
        action="store_true",
        help="Keep previous run outputs (default: auto-clean stale artifacts)",
    )
    p.add_argument(
        "--proj_only",
        action="store_true",
        help="Only extract projection coords (skip voxel / VLM / SAM3 / render). "
        "Requires <scene>/vlm_visual_results/localization_results.json",
    )
    return p.parse_args()

def main():
    args = parse_args()
    dtype = args.dataset_type
    scene = args.dataset_name
    base = os.path.abspath(require_nonempty(args.base_data_dir, "--base_data_dir"))
    iteration = args.iteration
    if iteration is None:
        iteration = 90000 if dtype in ("scannet", "gr_scannet") else 30000

    gs_py = os.path.abspath(require_nonempty(args.gs_python, "--gs_python"))
    if not os.path.isfile(gs_py):
        raise SystemExit(f"[Pipeline] gs_python not found: {gs_py}")

    needs_vlm = not args.proj_only and not args.skip_mask
    if needs_vlm:
        require_vlm_endpoint(args.api_key, args.base_url, args.model_name)

    print(f"[Pipeline] type={dtype} scene={scene}")
    print(f"[Pipeline] base_data_dir={base} iteration={iteration}")
    print(f"[Pipeline] gs_python={gs_py}")

    if args.proj_only:
        if dtype not in ("grlerf", "Ref-lerf", "gr_scannet"):
            raise SystemExit(
                f"[Pipeline] --proj_only not applicable for dataset_type={dtype}"
            )
        print("[Pipeline] --proj_only: skipping voxel / VLM / SAM3 / render")
        _run(
            "mask_generator.py --proj_only",
            [
                gs_py,
                os.path.join(ROOT, "mask_generator.py"),
                "--dataset_type",
                dtype,
                "--dataset_name",
                scene,
                "--base_data_dir",
                base,
                "--iteration",
                str(iteration),
                "--proj_only",
            ]
            + (["--keep_artifacts"] if args.keep_artifacts else []),
        )
        print("[Pipeline] All requested stages finished.")
        return

    if needs_vlm:
        print(f"[Pipeline] VLM API base_url={args.base_url} model={args.model_name}")

    # ---- A. voxel_grid_main.py ----
    if not args.skip_voxel:
        if dtype in ("grlerf", "Ref-lerf", "gr_scannet"):
            cmd = [
                gs_py,
                os.path.join(ROOT, "voxel_grid_main.py"),
                "--dataset_type",
                dtype,
                "--dataset_name",
                scene,
                "--base_data_dir",
                base,
                "--iteration",
                str(iteration),
                "--voxel_size",
                str(args.voxel_size),
            ]
            if args.quiet:
                cmd.append("--quiet")
            _run("voxel_grid_main.py", cmd)
        else:
            print(f"[Pipeline] voxel not required for {dtype}, skip.")
    else:
        print("[Pipeline] --skip_voxel")

    # ---- B. mask: VLM API (+ SAM3) ----
    if not args.skip_mask:
        mask_script = os.path.join(ROOT, "mask_generator.py")
        cmd = [
            gs_py,
            mask_script,
            "--dataset_type",
            dtype,
            "--dataset_name",
            scene,
            "--base_data_dir",
            base,
            "--iteration",
            str(iteration),
            "--base_url",
            args.base_url,
            "--model_name",
            args.model_name,
            "--api_key",
            args.api_key,
        ]
        if args.keep_artifacts:
            cmd.append("--keep_artifacts")
        _run("mask_generator.py", cmd)
    else:
        print("[Pipeline] --skip_mask")

    # ---- C. render.py ----
    if not args.skip_render:
        cmd = [
            gs_py,
            os.path.join(ROOT, "render.py"),
            "--dataset_type",
            dtype,
            "--dataset_name",
            scene,
            "--base_data_dir",
            base,
            "--k",
            str(args.k),
            "--threshold",
            str(args.threshold),
        ]
        if args.skip_train:
            cmd.append("--skip_train")
        if args.skip_test:
            cmd.append("--skip_test")
        if args.quiet:
            cmd.append("--quiet")
        if args.keep_artifacts:
            cmd.append("--keep_artifacts")
        _run("render.py", cmd)
    else:
        print("[Pipeline] --skip_render")

    print("[Pipeline] All requested stages finished.")

if __name__ == "__main__":
    main()
