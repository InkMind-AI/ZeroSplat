import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sam3_package_path = os.path.join(ROOT, "sam3_main")
if sam3_package_path not in sys.path:
    sys.path.append(sam3_package_path)

from preprocess.pre_dataset import (
    dataset_dir_name,
    get_json_names,
    get_prompt_render,
    select_test_image,
)

def _sanitize_category_dir(category):
    return str(category).replace("/", "_").replace("\\", "_").strip() or "default"

def merge_masks_across_categories(output_dir_prompt, category_subdirs, frame_files):
    """
    Merge per-category masks for the same referring expression via Logical OR.
    Args:
        output_dir_prompt: Root output directory for this referring expression.
        category_subdirs: Mapping from category name to its mask subdirectory.
        frame_files: Frame filenames used to resolve frame basenames.
    """
    frame_bases = {os.path.splitext(f)[0] for f in frame_files}
    all_frames_with_mask = set()
    for cat_dir in category_subdirs.values():
        if not os.path.isdir(cat_dir):
            continue
        for f in os.listdir(cat_dir):
            if f.endswith(".png") and not f.startswith("conf_") and "merged" not in f:
                base = os.path.splitext(f)[0]
                if base in frame_bases:
                    all_frames_with_mask.add(base)

    if not all_frames_with_mask:
        print("[Merge] No valid masks to merge")
        return

    os.makedirs(output_dir_prompt, exist_ok=True)
    for frame_base in sorted(all_frames_with_mask):
        mask_path = os.path.join(output_dir_prompt, frame_base + ".png")
        merged_mask = None
        for cat_dir in category_subdirs.values():
            p = os.path.join(cat_dir, frame_base + ".png")
            if not os.path.isfile(p):
                continue
            try:
                img = Image.open(p).convert("L")
                arr = (np.array(img) > 0).astype(np.float32)
            except Exception as e:
                print(f"[Merge] Failed to read {p}: {e}")
                continue
            if merged_mask is None:
                merged_mask = arr.copy()
            else:
                merged_mask = np.maximum(merged_mask, arr)

        if merged_mask is not None:
            out_img = Image.fromarray((merged_mask * 255).astype(np.uint8), mode="L")
            out_img.save(mask_path)
            print(f"[Merge] Merged frame {frame_base}.png -> {output_dir_prompt}")

def batch_segment_frames(
    input_dir,
    output_dir,
    text_prompt_list={"all objects"},
    text_json={},
    exclude_image_list=[],
    dataset_type="",
):
    """Batch-segment frames with SAM3 for Ref-lerf / gr_scannet / lerf / scannet."""
    import torch
    from filter_masks import extract_masks_and_scores, filter_masks_by_confidence
    from sam3_main.sam3.model.sam3_image_processor import Sam3Processor
    from sam3_main.sam3.model_builder import build_sam3_image_model

    dataset_type = str(dataset_type or "").strip()
    print(f"Skipping {len(exclude_image_list)} images. (dataset_type={dataset_type})")

    os.makedirs(output_dir, exist_ok=True)

    print("Loading SAM3 model...")
    sam3_ckpt = os.path.join(ROOT, "sam3_main", "sam3.pt")
    if os.path.isfile(sam3_ckpt):
        model = build_sam3_image_model(checkpoint_path=sam3_ckpt, load_from_HF=False)
    else:
        model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=0.0000001)
    print("Model loaded successfully.")

    all_frame_files = sorted(
        [f for f in os.listdir(input_dir) if f.endswith((".jpg", ".png"))]
    )
    if not all_frame_files:
        print("No image frames found in input directory!")
        return

    exclude_set = frozenset(exclude_image_list)
    frame_files = [
        f
        for f in all_frame_files
        if (os.path.splitext(f)[0] not in exclude_set) and (f not in exclude_set)
    ]
    skipped_count = len(all_frame_files) - len(frame_files)

    for prompt_idx, text_prompt in enumerate(text_prompt_list):
        output_dir_prompt = os.path.join(output_dir, f"{text_prompt}")
        os.makedirs(output_dir_prompt, exist_ok=True)
        print(
            f"\n========== Processing prompt {prompt_idx + 1}/{len(text_prompt_list)}: "
            f"{text_prompt}=========="
        )

        if dataset_type in ["Ref-lerf", "gr_scannet", "grlerf"]:
            prompt_val = text_json.get(text_prompt, text_prompt)
            category_list = prompt_val if isinstance(prompt_val, list) else [prompt_val]

            for cat_idx, category in enumerate(category_list):
                cat_safe = _sanitize_category_dir(category)
                output_dir_category = os.path.join(output_dir_prompt, cat_safe)
                os.makedirs(output_dir_category, exist_ok=True)
                print(f"  --- Category {cat_idx + 1}/{len(category_list)}: {category} ---")

                mask_score_dict = {}
                mask_second_score_dict = {}
                for frame_file in frame_files:
                    frame_path = os.path.join(input_dir, frame_file)
                    image = Image.open(frame_path).convert("RGB")
                    with torch.inference_mode(), torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16
                    ):
                        inference_state = processor.set_image(image)
                        output = processor.set_text_prompt(
                            prompt=category, state=inference_state
                        )
                        extract_masks_and_scores(
                            output, frame_file, mask_score_dict, mask_second_score_dict
                        )

                filter_masks_by_confidence(
                    mask_score_dict,
                    mask_second_score_dict,
                    output_dir_category,
                    threshold1=0.6,
                    threshold2=0.3,
                    second_threshold=0.8,
                    th1=0.15,
                )

            category_dirs = {
                cat: os.path.join(output_dir_prompt, _sanitize_category_dir(cat))
                for cat in category_list
            }
            merge_masks_across_categories(output_dir_prompt, category_dirs, frame_files)

        elif dataset_type in ["lerf", "scannet"]:
            mask_score_dict = {}
            mask_second_score_dict = {}
            for frame_file in frame_files:
                frame_path = os.path.join(input_dir, frame_file)
                image = Image.open(frame_path).convert("RGB")
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    inference_state = processor.set_image(image)
                    output = processor.set_text_prompt(
                        prompt=text_prompt, state=inference_state
                    )
                    extract_masks_and_scores(
                        output, frame_file, mask_score_dict, mask_second_score_dict
                    )

            filter_masks_by_confidence(
                mask_score_dict,
                mask_second_score_dict,
                output_dir_prompt,
                threshold1=0.6,
                threshold2=0.3,
                second_threshold=0.8,
                th1=0,
            )
        else:
            print(
                f"Invalid dataset_type='{dataset_type}'. "
                f"Expected one of: Ref-lerf, gr_scannet, lerf, scannet"
            )
            sys.exit(1)

if __name__ == "__main__":
    from utils.required_cli import require_nonempty, require_vlm_endpoint

    parser = argparse.ArgumentParser(description="SAM3 mask generation (+ VLM API)")

    # --- dataset ---
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="grlerf",
        choices=["Ref-lerf", "scannet", "lerf", "gr_scannet", "grlerf"],
    )
    parser.add_argument("--dataset_name", type=str, default="ramen", help="Scene folder name")
    parser.add_argument("--base_data_dir", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=30000)

    # --- VLM API (OpenAI-compatible, e.g. DashScope) ---
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key (required unless --skip_vlm / --proj_only). Local vLLM: EMPTY.",
    )
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument(
        "--vlm_only",
        action="store_true",
        help="Run only the VLM stage (write llm_ayal.json), then exit",
    )
    parser.add_argument(
        "--skip_vlm",
        action="store_true",
        help="Skip VLM; load category JSON and run SAM3 (+ projection)",
    )
    parser.add_argument(
        "--proj_only",
        action="store_true",
        help="Skip VLM/SAM3; only extract projection coords from localization_results.json",
    )
    parser.add_argument(
        "--keep_artifacts",
        action="store_true",
        help="Keep previous sam3_mask / VLM outputs (default: clean before run)",
    )

    # --- paths (optional; auto-derived from base_data_dir if empty) ---
    parser.add_argument(
        "--vis_save_root",
        type=str,
        default="",
        help="Default: <base>/<type>/<scene>/vlm_visual_results",
    )

    args = parser.parse_args()
    if sum(bool(x) for x in (args.vlm_only, args.skip_vlm, args.proj_only)) > 1:
        raise SystemExit("Use only one of --vlm_only / --skip_vlm / --proj_only")

    needs_vlm = not args.skip_vlm and not args.proj_only
    if needs_vlm:
        require_vlm_endpoint(args.api_key, args.base_url, args.model_name)

    DATASET_NAME = args.dataset_name
    DATASET_TYPE = args.dataset_type
    BASE_DATA_DIR = os.path.abspath(args.base_data_dir)
    scene_dir = os.path.join(BASE_DATA_DIR, dataset_dir_name(DATASET_TYPE), DATASET_NAME)
    vis_save_root = args.vis_save_root or os.path.join(scene_dir, "vlm_visual_results")

    if not args.keep_artifacts:
        from utils.run_artifacts import clean_mask_stage

        clean_mask_stage(
            scene_dir,
            vis_save_root,
            vlm_only=args.vlm_only,
            skip_vlm=args.skip_vlm,
            proj_only=args.proj_only,
        )

    def _run_projection():
        if DATASET_TYPE not in ["Ref-lerf", "gr_scannet", "grlerf"]:
            raise SystemExit(
                f"[proj] projection not applicable for dataset_type={DATASET_TYPE}"
            )
        from coord_extractor import get_pro_xy_main

        image_path = os.path.join(vis_save_root, "localization_results.json")
        if not os.path.isfile(image_path):
            raise SystemExit(f"[proj] missing localization JSON: {image_path}")
        proj_dict_dir = os.path.join(vis_save_root, "proj.pth")

        print("\n" + "=" * 60)
        print("Starting projection coordinate extraction")
        print("=" * 60)

        get_pro_xy_main(
            dataset_type=DATASET_TYPE,
            dataset_name=DATASET_NAME,
            model_path=scene_dir,
            source_path=scene_dir,
            image_path=image_path,
            proj_dict_dir=proj_dict_dir,
            iteration=args.iteration,
            quiet=False,
        )
        print(f"Projection coordinates saved to: {proj_dict_dir}\n")

    if args.proj_only:
        _run_projection()
        sys.exit(0)

    args.test_image_names = select_test_image(args.dataset_type, args.dataset_name)
    args.target_name = get_prompt_render(
        args.dataset_type, args.dataset_name, BASE_DATA_DIR
    )
    args.json_names = get_json_names(DATASET_TYPE, DATASET_NAME)
    save_json_path = os.path.join(scene_dir, "llm_ayal.json")

    MODEL_NAME = args.model_name
    print(f"[VLM] base_url={args.base_url}, model={MODEL_NAME}")

    OUTPUT_MASKS_DIR = os.path.join(scene_dir, "sam3_mask")
    category_results = {}
    text_prompt_list = []
    EXCLUDE_LIST = []
    INPUT_FRAMES_DIR = os.path.join(scene_dir, "images")

    def _run_vlm(json_path, json_names, frame_choice_path):
        from preprocess.pre_data_llm import data_process_llm

        return data_process_llm(
            json_path=json_path,
            json_names=json_names,
            MODEL_NAME=MODEL_NAME,
            frame_choice_path=frame_choice_path,
            save_json_path=save_json_path,
            vis_save_root=vis_save_root,
            dataset_type=DATASET_TYPE,
            api_key=args.api_key,
            base_url=args.base_url,
        )

    def _load_category_results():
        if not os.path.isfile(save_json_path):
            raise SystemExit(
                f"[skip_vlm] missing category JSON: {save_json_path}. "
                "Run VLM stage first (--vlm_only)."
            )
        with open(save_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[skip_vlm] loaded categories from {save_json_path} ({len(data)} keys)")
        return data

    if DATASET_TYPE in ["Ref-lerf", "lerf", "grlerf"]:
        INPUT_FRAMES_DIR = os.path.join(scene_dir, "images")
        json_names = args.json_names
        EXCLUDE_LIST = [name.replace(".json", "") for name in json_names]
        print(
            f"Detected dataset type '{DATASET_TYPE}', enabling exclude-list filtering "
            f"(if_reflerfovs=True)"
        )

        if DATASET_TYPE == "Ref-lerf":
            json_path = os.path.join(scene_dir, "json", "test_json")
        elif DATASET_TYPE == "grlerf":
            json_path = os.path.join(scene_dir, "json")
        else:
            json_path = None

        if DATASET_TYPE in ["Ref-lerf", "grlerf"]:
            frame_choice_path = os.path.join(scene_dir, "scannet_weighted_frames.json")
            if args.skip_vlm:
                category_results = _load_category_results()
            else:
                category_results = _run_vlm(json_path, json_names, frame_choice_path)
            text_prompt_list = list(category_results.keys())
        elif DATASET_TYPE == "lerf":
            text_prompt_list = args.target_name
        else:
            print("Please provide a valid dataset type")
            sys.exit(1)

    elif DATASET_TYPE == "scannet":
        INPUT_FRAMES_DIR = os.path.join(scene_dir, "color")
        EXCLUDE_LIST = []
        print(
            f"Detected dataset type '{DATASET_TYPE}', disabling exclude-list filtering "
            f"(if_reflerfovs=False)"
        )
        text_prompt_list = args.target_name

    elif DATASET_TYPE == "gr_scannet":
        INPUT_FRAMES_DIR = os.path.join(scene_dir, "color")
        EXCLUDE_LIST = []
        frame_choice_path = os.path.join(scene_dir, "scannet_weighted_frames.json")
        json_path = os.path.join(scene_dir, f"{DATASET_NAME}_text")
        json_names = args.json_names
        if args.skip_vlm:
            category_results = _load_category_results()
        else:
            category_results = _run_vlm(json_path, json_names, frame_choice_path)
        text_prompt_list = list(category_results.keys())
    else:
        print("Please provide a valid dataset type")
        sys.exit(1)

    if args.vlm_only:
        print("[vlm_only] VLM stage finished; skipping SAM3 / projection.")
        sys.exit(0)

    batch_segment_frames(
        input_dir=INPUT_FRAMES_DIR,
        output_dir=OUTPUT_MASKS_DIR,
        text_prompt_list=text_prompt_list,
        text_json=category_results,
        exclude_image_list=EXCLUDE_LIST,
        dataset_type=DATASET_TYPE,
    )

    if DATASET_TYPE in ["Ref-lerf", "gr_scannet", "grlerf"]:
        _run_projection()
