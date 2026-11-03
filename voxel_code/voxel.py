import os
import json
import torch
import numpy as np
import struct
import collections
import open3d as o3d
from PIL import Image

Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_intrinsics_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            if model_id == 0: num_params = 3; model_name = "SIMPLE_PINHOLE"
            elif model_id == 1: num_params = 4; model_name = "PINHOLE"
            else: num_params = 0; model_name = "UNKNOWN"
            params = read_next_bytes(fid, num_bytes=8*num_params, format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id, model=model_name, width=width, height=height, params=np.array(params))
    return cameras

def read_extrinsics_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            fid.read(num_points2D * 24)
            R = qvec2rotmat(qvec)
            t = tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)
            c2w = np.linalg.inv(w2c)
            images[image_id] = {"name": image_name, "camera_id": camera_id, "c2w": c2w}
    return sorted(images.values(), key=lambda x: x["name"])

def unproject(intrinsics, poses, depths):
    V, H, W = depths.shape
    y, x = torch.meshgrid(torch.arange(0, H).to(depths.device), torch.arange(0, W).to(depths.device), indexing='ij')
    x = x.unsqueeze(0).repeat(V, 1, 1).view(V, H*W)
    y = y.unsqueeze(0).repeat(V, 1, 1).view(V, H*W)
    fx = intrinsics[:, 0, 0].unsqueeze(-1).repeat(1, H*W)
    fy = intrinsics[:, 1, 1].unsqueeze(-1).repeat(1, H*W)
    cx = intrinsics[:, 0, 2].unsqueeze(-1).repeat(1, H*W)
    cy = intrinsics[:, 1, 2].unsqueeze(-1).repeat(1, H*W)
    z = depths.view(V, H*W) / 1000.0 
    
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    cam_coords = torch.stack([x, y, z, torch.ones_like(x)], -1)
    world_coords = (poses @ cam_coords.permute(0, 2, 1)).permute(0, 2, 1)
    return world_coords[..., :3] / world_coords[..., 3].unsqueeze(-1)

def compute_curvature(points, radius=0.1):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    curvatures = []
    num_points = len(points)

    for i in range(num_points):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], radius)
        if k < 4: 
            curvatures.append(0.0)
            continue
        neighbors = points[idx, :]
        cov = np.cov(neighbors.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        curvature = eigenvalues[0] / (np.sum(eigenvalues) + 1e-6)
        curvatures.append(curvature)
        
    return np.array(curvatures)

class VideoProcessor:
    def __init__(self, video_folder="", voxel_size=0.1, min_xyz_range=None, max_xyz_range=None):
        self.video_folder = video_folder
        self.voxel_size = voxel_size
        self.min_xyz_range = torch.tensor(min_xyz_range) if min_xyz_range is not None else None
        self.max_xyz_range = torch.tensor(max_xyz_range) if max_xyz_range is not None else None
        self.scene = {}
        self.default_scene_id = None

        print("VideoProcessor Initialized.")

        if self.video_folder:
            scene_root = os.path.dirname(self.video_folder.rstrip('/')) 
            scene_id = os.path.basename(scene_root)
            sparse_dir = os.path.join(scene_root, "sparse", "0")
            transforms_train_path = os.path.join(scene_root, "transforms_train.json")
            
            if os.path.exists(sparse_dir) and os.path.exists(os.path.join(sparse_dir, "cameras.bin")):
                print(f"Found COLMAP binary model at: {sparse_dir}")
                cameras = read_intrinsics_binary(os.path.join(sparse_dir, "cameras.bin"))
                images_data = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
                
                image_list = []
                first_cam = cameras[list(cameras.keys())[0]]
                K = np.eye(3)
                if first_cam.model == "PINHOLE": 
                    K[0,0], K[1,1], K[0,2], K[1,2] = first_cam.params
                elif first_cam.model == "SIMPLE_PINHOLE":
                    K[0,0], K[1,1], K[0,2], K[1,2] = first_cam.params[0], first_cam.params[0], first_cam.params[1], first_cam.params[2]
                
                for img in images_data:
                    image_list.append({"img_path": img["name"], "pose": img["c2w"].tolist()})
                
                self.scene[scene_id] = {"images": image_list, "depth_cam2img": K.tolist(), "axis_align_matrix": np.eye(4).tolist()}
                self.default_scene_id = scene_id
                print(f"[*] Successfully loaded {len(image_list)} frames for scene '{scene_id}'")
            
            elif os.path.exists(transforms_train_path):
                print(f"[*] Found ScanNet transforms at: {transforms_train_path}")
                with open(transforms_train_path, 'r') as f:
                    transforms_data = json.load(f)
                
                w = transforms_data["w"]
                h = transforms_data["h"]
                fl_x = transforms_data["fl_x"]
                fl_y = transforms_data["fl_y"]
                cx = transforms_data.get("cx", w / 2.0)
                cy = transforms_data.get("cy", h / 2.0)
                
                K = np.array([
                    [fl_x, 0, cx],
                    [0, fl_y, cy],
                    [0, 0, 1]
                ])
                
                image_list = []
                for frame in transforms_data["frames"]:
                    file_path = frame["file_path"]
                    img_name = os.path.basename(file_path) + ".jpg"
                    c2w_raw = np.array(frame["transform_matrix"])
                    c2w = c2w_raw.copy()
                    c2w[:3, 1:3] *= -1
                    
                    image_list.append({"img_path": img_name, "pose": c2w.tolist()})
                
                self.scene[scene_id] = {"images": image_list, "depth_cam2img": K.tolist(), "axis_align_matrix": np.eye(4).tolist()}
                self.default_scene_id = scene_id
                print(f"Successfully loaded {len(image_list)} frames for ScanNet scene '{scene_id}'")
            
            else:
                print(f"Warning: Neither COLMAP sparse/0 nor transforms_train.json found at {scene_root}")

    def calculate_world_coords(self, video_id, frame_files):
        if video_id not in self.scene and self.default_scene_id: video_id = self.default_scene_id
        if video_id not in self.scene: return {}
        
        meta = self.scene[video_id]
        K = torch.tensor(meta["depth_cam2img"])
        axis_align = torch.tensor(meta["axis_align_matrix"])
        name2pose = {img['img_path']: img.get('pose') for img in meta['images']}
        
        depths, poses = [], []
        valid_frames = []
        scene_root = os.path.dirname(self.video_folder.rstrip(os.sep))
        for f in frame_files:
            name = os.path.basename(f)
            name_no_ext = os.path.splitext(name)[0]
            d_path = os.path.join(scene_root, "depth_image", name_no_ext + ".png")
            if os.path.exists(d_path) and name in name2pose:
                depths.append(torch.from_numpy(np.array(Image.open(d_path)).astype(np.uint16)))
                poses.append(torch.from_numpy(np.array(name2pose[name])))
                valid_frames.append(f)
        
        if not depths: return {"world_coords": torch.empty(0)}
        
        depths = torch.stack(depths)
        poses = torch.stack([axis_align.float() @ p.float() for p in poses])
        K_batch = K.unsqueeze(0).repeat(len(depths), 1, 1)
        
        world_coords = unproject(K_batch.float(), poses.float(), depths.float())
        return {"world_coords": world_coords, "valid_frames": valid_frames}

    def discrete_point(self, xyz, return_tensor=False): 
        if isinstance(xyz, np.ndarray): xyz = torch.from_numpy(xyz)
        
        if not isinstance(xyz, torch.Tensor):
            xyz = torch.tensor(xyz)
            
        xyz = xyz.float()
        
        if self.min_xyz_range is not None: 
            xyz = torch.max(xyz, self.min_xyz_range.to(xyz.device))
        if self.max_xyz_range is not None: 
            xyz = torch.min(xyz, self.max_xyz_range.to(xyz.device))
        if self.min_xyz_range is not None: 
            xyz = xyz - self.min_xyz_range.to(xyz.device)
            
        result = (xyz / self.voxel_size).round().int()
        
        if return_tensor:
            return result
            
        return result.tolist()

def generate_maps_from_ply(ply_path, processor):
   
    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        if len(points) == 0:
            print("[!] Error: Point cloud is empty.")
            return None, None
        
        print(f"    Raw Points: {len(points)}")
        
        voxel_coords_list = processor.discrete_point(points)  
        unique_voxels = list(set([tuple(v) for v in voxel_coords_list]))
        print(f"Unique Voxels: {len(unique_voxels)}")
        
        print("Computing geometric curvature...")
        pcd_down = pcd.voxel_down_sample(voxel_size=processor.voxel_size * 0.5)
        points_down = np.asarray(pcd_down.points)
        
        search_radius = processor.voxel_size * 2.5
        curvatures = compute_curvature(points_down, radius=search_radius)
        
        weights = []
        if len(curvatures) > 0:
            c_min, c_max = np.min(curvatures), np.max(curvatures)
            weights = 1.0 + 9.0 * (curvatures - c_min) / (c_max - c_min + 1e-6)
        else:
            weights = np.ones(len(points_down))
            
        voxel_coords_down = processor.discrete_point(points_down)
        
        scene_voxel_weights = {}
        for coord, w in zip(voxel_coords_down, weights):
            coord_tuple = tuple(coord)
            if coord_tuple not in scene_voxel_weights:
                scene_voxel_weights[coord_tuple] = float(w)
            else:
                scene_voxel_weights[coord_tuple] = max(scene_voxel_weights[coord_tuple], float(w))
        
        print(f"    Weighted Voxels: {len(scene_voxel_weights)}")
        return unique_voxels, scene_voxel_weights
    except Exception as e:
        print(f"[!] Error processing PLY: {e}")
        return None, None

def voxel_grid_main(args):
    base_dir = os.path.join(args.video_dir, args.scene_name)
    images_dir = getattr(args, "images_dir", None) or os.path.join(base_dir, "images")
    ply_path = getattr(args, "ply_path", None) or os.path.join(base_dir, "point_cloud/iteration_30000/point_cloud.ply")
    video_processor = VideoProcessor(
        video_folder=images_dir,
        voxel_size=args.voxel_size,
    )
    
    if not video_processor.scene:
        print("Failed to initialize VideoProcessor.")
        return
    
    target_scene_id = video_processor.default_scene_id
    pc_voxel_set = set()
    scene_weights = {}
    
    if not os.path.exists(ply_path):
        print(f"Error: PLY file not found at {ply_path}")
        return
            
    unique_voxels, generated_weights = generate_maps_from_ply(ply_path, video_processor)
        
    if unique_voxels and generated_weights:
        pc_voxel_set = set(tuple(v) for v in unique_voxels)
        scene_weights = generated_weights
    else:
        return

    meta_info = video_processor.scene[target_scene_id]
    frame_files = [os.path.join(video_processor.video_folder, img["img_path"]) for img in meta_info["images"]]
    print(f"Calculating 3D coverage for {len(frame_files)} frames...")
    world_coords_res = video_processor.calculate_world_coords(target_scene_id, frame_files)
    world_coords = world_coords_res.get("world_coords")

    valid_frames = world_coords_res.get("valid_frames")
    if valid_frames is not None:
        print(f"    -> Valid frames: {len(valid_frames)} (dropped {len(frame_files) - len(valid_frames)})")
        frame_files = valid_frames

    if world_coords is None or len(world_coords) == 0:
        print("Error: Failed to calculate world coords.")
        return

    world_coords = world_coords.to('cuda')
    world_coords_discrete_tensor = video_processor.discrete_point(world_coords,return_tensor=True)
    frame_voxel_map = {}
    for i, frame_file in enumerate(frame_files):
        coords = world_coords_discrete_tensor[i].view(-1, 3)
        coords = torch.unique(coords, dim=0).cpu().tolist()
        frame_voxel_map[frame_file] = set(tuple(c) for c in coords)

    candidate_frames = list(frame_files)
    print("Starting Greedy WGSC Sampling...")
    select_frame_files = []
    used_voxels = set()
    accumulated_weights = []
    max_frames = 30

    for _ in range(max_frames):
        best_frame = None
        max_gain = -1.0

        for frame in candidate_frames:
            current_frame_voxels = frame_voxel_map[frame]
            valid_voxels = current_frame_voxels & pc_voxel_set
            new_voxels = valid_voxels - used_voxels
            if not new_voxels: continue
            gain = sum(scene_weights.get(v, 1.0) for v in new_voxels)
            if gain > max_gain:
                max_gain = gain
                best_frame = frame

        if best_frame is None or max_gain <= 0:
            break

        select_frame_files.append(best_frame)
        candidate_frames.remove(best_frame)
        used_voxels.update(frame_voxel_map[best_frame] & pc_voxel_set)
        accumulated_weights.append(max_gain)
        print(f"   -> Selected: {os.path.basename(best_frame)} | Gain: {max_gain:.2f}")

    output_data = [{
        "video_id": target_scene_id,
        "frame_files": select_frame_files,
        "salience_weights": accumulated_weights,
        "num_select_frames": len(select_frame_files),
    }]
    
    if not os.path.exists(args.output_filename):
        os.makedirs(os.path.dirname(args.output_filename), exist_ok=True)

    with open(args.output_filename, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Result saved to {args.output_filename}")
