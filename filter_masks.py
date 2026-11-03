import torch
import numpy as np
import os
from PIL import Image
import sys

def _to_numpy(x):
    """NumPy does not support bfloat16; SAM3 under autocast returns bf16 floats."""
    if not isinstance(x, torch.Tensor):
        return x
    if x.is_floating_point():
        return x.float().cpu().numpy()
    return x.cpu().numpy()

def extract_mask_and_score_max(output, frame_file):
    scores = output.get('scores')
    masks = output.get('masks')
    
    if scores is None or len(scores) == 0:
        output['scores'], output['masks'] = [], []
        return output
    
    if masks.ndim == 4:
        if masks.shape[1] == 1:
            masks = masks.squeeze(1)  
        else:
            raise ValueError(
                f"4D mask expected dim1==1, got shape {masks.shape}"
            )
    
    max_idx = torch.argmax(scores).item()
    
    max_score = scores[max_idx].view(1) 
    max_mask = masks[max_idx].unsqueeze(0) 

    output['scores'] = max_score
    output['masks'] = max_mask

    return output

def extract_masks_and_scores(output, frame_file, mask_score_dict,mask_second_score_dict):
        masks = output.get("masks", None)
        scores = output.get("scores", None)  
        
        if masks is None or masks.numel() == 0:
            print(f"No masks found for {frame_file}, skipping.")
            mask_score_dict[frame_file] = {"masks": None, "scores": None}
            return
        if scores is None or scores.numel() == 0:
            print(f"No confidence scores found for {frame_file}, skipping.")
            mask_score_dict[frame_file] = {"masks": masks, "scores": None}
            return
        
        if len(scores) >= 2:
            _, top_indices = torch.topk(scores, k=2)
            sec_idx = top_indices[1].item()
            mask_sec = masks[sec_idx].unsqueeze(0)
            score_sec = scores[sec_idx].view(1)
        else:
            print("No second-highest confidence mask available")
            sys.exit(1)
        mask_second_score_dict[frame_file] = {
            "masks": _to_numpy(mask_sec),
            "scores": _to_numpy(score_sec),
        }
        output = extract_mask_and_score_max(output,frame_file)

        masks = output.get("masks", None)
        scores = output.get("scores", None)

        if masks is None  or len(masks)==0:
            print(f"No masks found for {frame_file}, skipping.")
            mask_score_dict[frame_file] = {"masks": None, "scores": None}
            return
        if scores is None  or len(scores)==0:
            print(f"No confidence scores found for {frame_file}, skipping.")
            mask_score_dict[frame_file] = {"masks": masks, "scores": None}
            return
        
        mask_score_dict[frame_file] = {
            "masks": _to_numpy(masks),
            "scores": _to_numpy(scores),
        }

def save_second_masks(final_dict, valid_second_dict,output_dir,second_threshold=0.8):
    second_mask_dir = os.path.join(output_dir, "second_masks")
    merged_mask_dir = os.path.join(output_dir, "merged_high_conf") 
    os.makedirs(second_mask_dir, exist_ok=True)
    os.makedirs(merged_mask_dir, exist_ok=True)

    for frame_file in final_dict.keys():
        if frame_file not in valid_second_dict:
            continue
            
        content = valid_second_dict[frame_file]
        masks_second = content.get("masks")
        scores_second = content.get("scores")
        if isinstance(scores_second, torch.Tensor):
            score_val = scores_second[0].cpu().item()
        else:
            score_val = float(scores_second[0])
            
        if isinstance(masks_second, torch.Tensor):
            mask_data = _to_numpy(masks_second[0])
        else:
            mask_data = masks_second[0]
        mask_2d = np.squeeze(mask_data)
        if len(mask_2d.shape) != 2:
            continue

        mask_8bit = (mask_2d * 255).astype(np.uint8)
        if np.max(mask_8bit) == 0:
            continue
        if score_val > second_threshold:
            file_base = os.path.splitext(frame_file)[0]
            new_file_name = f"conf_{score_val:.4f}_{file_base}.png"
            mask_save_path = os.path.join(second_mask_dir, new_file_name)
            mask_image = Image.fromarray(mask_8bit, mode='L')
            mask_image.save(mask_save_path)
            print(f"Saved Second Mask: {new_file_name}")
            content_max = final_dict[frame_file]
            masks_max = content_max.get("masks")
            
            if masks_max is not None:
                if isinstance(masks_max, torch.Tensor):
                    mask_data_max = _to_numpy(masks_max[0])
                else:
                    mask_data_max = masks_max[0]
                
                mask_2d_max = np.squeeze(mask_data_max)
                
                merged_mask = np.maximum(mask_2d_max, mask_2d)
                new_mask_tensor = torch.from_numpy(merged_mask).unsqueeze(0)
                final_dict[frame_file]["masks"] = new_mask_tensor
                merged_8bit = (merged_mask * 255).astype(np.uint8)
                
                merged_name = f"merged_conf_{score_val:.4f}_{file_base}.png"
                merged_path = os.path.join(merged_mask_dir, merged_name)
                Image.fromarray(merged_8bit, mode='L').save(merged_path)
                print(f"[Merge] Merged top and second masks (score:{score_val:.4f}) -> {merged_name}")

def save_masks(valid_dict, output_dir): 
    saved_mask_num = 0  
    
    for frame_file, content in valid_dict.items():
        masks = content["masks"]
        scores = content["scores"]
        
        if isinstance(scores, torch.Tensor):
            best_score = scores[0].cpu().item()
        else:  
            best_score = float(scores[0])
        if isinstance(masks, torch.Tensor):
            best_mask = _to_numpy(masks[0])
        else:  
            best_mask = masks[0]
        best_mask = np.squeeze(best_mask)
        
        if len(best_mask.shape) != 2:
            print(f"Warning: Mask shape {best_mask.shape} is not 2D, skipping {frame_file}")
            continue

        mask_8bit = (best_mask * 255).astype(np.uint8)
        if np.max(mask_8bit) == 0: 
            print(f"Skipping {frame_file}: Mask is all black (empty). [Original Score: {best_score:.4f}]")
            continue
        
        mask_save_path = os.path.join(output_dir, os.path.splitext(frame_file)[0] + ".png")
        mask_image = Image.fromarray(mask_8bit, mode='L')
        mask_image.save(mask_save_path)
        
        saved_mask_num += 1
        print(f"Saved {frame_file} | Score: {best_score:.4f}")
    
    print(f"\nAll valid masks saved. Total: {saved_mask_num}")
    return saved_mask_num

def get_first_score(item):
    content = item[1]
    scores = content["scores"]
    if isinstance(scores, torch.Tensor):
        first_score = scores[0].cpu().item()
    else:
        first_score = float(scores[0])
    return first_score

def filter_masks_by_confidence(
    mask_score_dict,
    mask_second_score_dict,
    output_dir,
    threshold1=0.6,
    threshold2=0.3,
    second_threshold=0.8,
    th1=0,
    expand_k=30,
    low_conf_max_save=6,
):
    
    print(
        f"threshold1: {threshold1}, threshold2: {threshold2}, second_threshold: {second_threshold}, "
        f"th1: {th1}, expand_k: {expand_k}, low_conf_max_save: {low_conf_max_save}"
    )
    valid_dict = {
        img_name: content
        for img_name, content in mask_score_dict.items()
        if content["masks"] is not None and content["scores"] is not None
    }
    valid_second_dict = {
        img_name: content
        for img_name, content in mask_second_score_dict.items()
        if content["masks"] is not None and content["scores"] is not None
    }
    
    scored_items = [(get_first_score(item), item) for item in valid_dict.items()]
    sorted_items = sorted(scored_items, key=lambda x: x[0], reverse=True)
    valid_image_num = len(sorted_items)

    high_conf_items = [item for score, item in sorted_items if score > threshold2]
    super_conf_items = [item for score, item in sorted_items if score > threshold1]
    final_dict = dict(super_conf_items)

    if len(final_dict) < expand_k:
        if len(high_conf_items) > 0:
            num_to_save = min(len(high_conf_items), expand_k)
            print("N_target:",num_to_save)
            print(f"Found {len(high_conf_items)} images with confidence > {threshold2}, saving top {num_to_save} masks")
            final_dict = dict(high_conf_items[:num_to_save])
        else:
            low_conf_threshold = th1
            above_015_items = [(s, item) for s, item in sorted_items if s > low_conf_threshold]
            if len(above_015_items) == 0:
                final_dict = {}
                print(f"No images with confidence > {threshold2}, and none with confidence > {low_conf_threshold}; saving no masks")
            else:
                num_to_save = min(len(above_015_items), int(low_conf_max_save))
                final_dict = dict(item for _, item in above_015_items[:num_to_save])
                print(f"No images with confidence > {threshold2}; saving top {num_to_save} with confidence > {low_conf_threshold} (total {len(above_015_items)} > {low_conf_threshold}, cap={low_conf_max_save})")
   
    save_second_masks(final_dict, valid_second_dict, output_dir, second_threshold=second_threshold)
    saved_mask_num = save_masks(final_dict, output_dir)
    
    print(f"Final number of saved masks: {saved_mask_num}")
    
    return valid_dict, saved_mask_num
