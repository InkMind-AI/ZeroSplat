import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import argparse
import sys
import os
from plyfile import PlyData

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from preprocess.pre_dataset import get_prompt_render

def refine_labels_knn(model_points, pred_mask_tensor, k=20, threshold=0.8):
    current_labels = (pred_mask_tensor == 1).cpu().numpy().astype(np.int8)

    nn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', n_jobs=-1)
    nn.fit(model_points)

    black_pts_indices = np.where(current_labels == 0)[0]

    if len(black_pts_indices) == 0:
        return pred_mask_tensor

    _, neighbors_idx = nn.kneighbors(model_points[black_pts_indices])

    neighbor_labels = current_labels[neighbors_idx]
    red_neighbor_counts = np.sum(neighbor_labels == 1, axis=1)

    to_fix_mask = (red_neighbor_counts / k) >= threshold
    indices_to_fix = black_pts_indices[to_fix_mask]

    refined_labels = current_labels.copy()
    refined_labels[indices_to_fix] = 1

    print(f"KNN spread: relabeled {len(indices_to_fix)} interior points as foreground")

    return torch.from_numpy(refined_labels == 1).to(pred_mask_tensor.device)

def knn_spread_main(pts_path="",label_path="",save_dir="",k=40,threshold=0.2,dataset_name="",dataset_type="",text=""):

    print("k=",k)
    print("threshold=",threshold)
    if pts_path.endswith('.ply'):
        ply_data = PlyData.read(pts_path)
        vertex = ply_data['vertex']
        points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    else:
        print("Unsupported Gaussian file format (expected .ply)")
        sys.exit(1)

    print(f"Loaded point cloud: {len(points)} points")
    category=text
    print(f"\n{'='*50}")
    print(f"Processing category: {category}")
    if dataset_type in ['Ref-lerf','lerf',"grlerf"]:
        relative_path = os.path.join(category, "train", "ours_30000", "label_refinement", "refined_labels.pth")
    elif dataset_type in ['scannet', 'gr_scannet']:
        relative_path = os.path.join(category, "train", "ours_90000", "label_refinement", "refined_labels.pth")
    else:
        print(f"Invalid dataset_type='{dataset_type}'; expected Ref-lerf, lerf, scannet, or gr_scannet")
        sys.exit(1)

    current_label_path = os.path.join(label_path, relative_path)

    print(f"Label path: {current_label_path}")

    if not os.path.exists(current_label_path):
        print(f"Skip: file not found {current_label_path}")

    try:
        refined_data = torch.load(current_label_path, map_location="cpu")
        raw_label_tensor = refined_data['refined_label']

        if len(points) != len(raw_label_tensor):
            print(f"Error: point count ({len(points)}) != label count ({len(raw_label_tensor)})")
            sys.exit(1)

        print(f"Running KNN repair (K={k}, T={threshold})...")
        fixed_mask = refine_labels_knn(points, raw_label_tensor, k=k, threshold=threshold)

        os.makedirs(save_dir, exist_ok=True)
        save_filename = f"{category}_fixed_labels_k{k}_t{threshold}.pth"
        save_full_path = os.path.join(save_dir, save_filename)

        refined_data['refined_label'] = fixed_mask.long().cpu()
        torch.save(refined_data, save_full_path)
        torch.save(refined_data, current_label_path)

        print(f"Saved: {save_full_path}")

    except Exception as e:
        print(f"Error processing {category}: {e}")

    print("\nKNN spread finished.")

def main():
    parser = argparse.ArgumentParser(description="Standalone 3D Gaussian semantic label KNN spread")
    parser.add_argument("--pts_path", type=str, required=True, help="Gaussian point cloud (.ply)")
    parser.add_argument("--label_path", type=str, required=True, help="Directory containing refined_labels.pth")
    parser.add_argument("--save_dir", type=str, required=True, help="Output directory")

    parser.add_argument("--k", type=int, default=40, help="KNN neighbor count")
    parser.add_argument("--threshold", type=float, default=0.2, help="Spread threshold (0.0-1.0)")

    parser.add_argument("--dataset_name", type=str, default="waldo_kitchen", help="Scene name")
    parser.add_argument("--dataset_type", type=str, default="reflerf", choices=["reflerf", "scannet", "lerf"], help="Dataset type")

    args = parser.parse_args()

    args.text = get_prompt_render(args.dataset_type,args.dataset_name)
    knn_spread_main(pts_path=args.pts_path, label_path=args.label_path, save_dir=args.save_dir, k=args.k, threshold=args.threshold, dataset_name=args.dataset_name, dataset_type=args.dataset_type, text=args.text)
