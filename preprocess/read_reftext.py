import json
import os
from pathlib import Path

# The released gr_scannet GT files are named "gr-scannet_<scene>.json"; older copies use "<scene>.json".
GR_SCANNET_GT_PREFIXES = ("", "gr-scannet_")

def resolve_gr_scannet_gt_path(base_dir, name):
    """Return the existing GT json path for `name` (e.g. 'scene0000_00.json') under `base_dir`.

    Tries "<name>" first, then "gr-scannet_<name>". Returns None if neither exists.
    """
    for prefix in GR_SCANNET_GT_PREFIXES:
        candidate = Path(base_dir) / f"{prefix}{name}"
        if candidate.is_file():
            return candidate
    return None

def batch_process_json(dataset_type,base_dir, file_names):
    final_result = {}

    for name in file_names:
        full_path = Path(base_dir) / name
        if dataset_type == "gr_scannet":
            resolved = resolve_gr_scannet_gt_path(base_dir, name)
            if resolved is None:
                # Do not skip silently: an empty prompt set would make the whole pipeline produce garbage.
                raise FileNotFoundError(
                    f"gr_scannet GT json not found: tried "
                    f"{[str(Path(base_dir) / (p + name)) for p in GR_SCANNET_GT_PREFIXES]}"
                )
            full_path = resolved
        if dataset_type in ["Ref-lerf","grlerf"]:
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                file_dict = {}
                objects = data.get("object", [])
                
                for obj in objects:
                    category = obj.get("category")
                    sentence = obj.get("sentence")
                    
                    if category is not None:
                        if category in file_dict:
                            raise ValueError(f"Duplicate category detected! File: {name}, category: '{category}'. Stopped.")
                        
                        file_dict[category] = sentence
                    
                final_result[name] = file_dict
                
            except FileNotFoundError:
                print(f"Warning: file not found {full_path}, skipped.")
        elif dataset_type == "gr_scannet":
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                file_dict = {}
                objs = data.get("gt_info", [])
                for obj in objs:
                    label_list = obj.get("label")
                    if not label_list or not isinstance(label_list, (list, tuple)):
                        continue
                    category_key = label_list[0]
                    if not category_key:
                        continue
                    if category_key in file_dict:
                        raise ValueError(f"Duplicate category detected! File: {name}, category: '{category_key}'. Stopped.")
                    file_dict[category_key] = label_list
                    
                final_result[name] = file_dict
                
            except FileNotFoundError:
                print(f"Warning: file not found {full_path}, skipped.")
                    

    return final_result
