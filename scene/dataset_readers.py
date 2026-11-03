#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import math
import os
import sys
import torch
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    objects: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, objects_folder,test_image_names):
    # cam_infos = []
    traincam_info = []
    testcam_info = []
    skipped_missing_image = []
    print("cam_extrinsics image count:", len(cam_extrinsics))
    print("cam_intrinsics image count:", len(cam_intrinsics))

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        # pdb.set_trace()

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "SIMPLE_RADIAL":
            focal_length = intr.params[0]
            FovY = focal2fov(focal_length, height)
            FovX = focal2fov(focal_length, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        # Load RGB frames for lift rasterization
        image_path = os.path.join(images_folder, os.path.basename(extr.name))

        if not os.path.isfile(image_path):
            skipped_missing_image.append(os.path.basename(extr.name))
            continue
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        
        object_path = os.path.join(objects_folder, image_name + '.png')

        if not os.path.exists(object_path):
            object_path = os.path.join(objects_folder, image_name + '.jpg')
        objects = Image.open(object_path) if os.path.exists(object_path) else None

        if objects is None:
            if image_name in test_image_names:

                # objects = Image.new('L', (width, height), 0) 
                pass 
            else:

                continue
        # here both R and T, are w2c.  --- convert to --> c2w
        
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, objects=objects)

        if image_name in test_image_names:
            testcam_info.append(cam_info)
            print(f"\n{image_name} added to test set")
        else:
            traincam_info.append(cam_info)
        # cam_infos.append(cam_info)
    sys.stdout.write('\n')
    if skipped_missing_image:
        print(f"[readColmapCameras] skipped {len(skipped_missing_image)} COLMAP entries with missing images "
              f"(folder: {images_folder}), e.g. {skipped_missing_image[:5]}")
    else:
        print("[readColmapCameras] no COLMAP entries skipped due to missing images")
    print(f"\nSplit done: train={len(traincam_info)}, test={len(testcam_info)}")
    return traincam_info, testcam_info
    # return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, object_path, test_image_names,target_name,llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    object_path = os.path.join(object_path,target_name)
    object_dir = 'object_mask' if object_path == None else object_path
    train_cam_infos_unsorted,test_cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
                                                            images_folder=os.path.join(path, reading_dir), 
                                                            objects_folder=os.path.join(path, object_dir),test_image_names=test_image_names)
    train_cam_infos = sorted(train_cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    test_cam_infos = sorted(test_cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        # train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        # test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        train_cam_infos = train_cam_infos
        test_cam_infos = test_cam_infos
    else:
        # train_cam_infos = cam_infos
        test_cam_infos = []

    if len(train_cam_infos) == 0:
        raise RuntimeError(
            f"No training views with masks for target under object masks "
            f"(test-only={len(test_cam_infos)}). Skip this target."
        )
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

# def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
#     cam_infos = []

#     with open(os.path.join(path, transformsfile)) as json_file:
#         contents = json.load(json_file)
#         fovx = contents["camera_angle_x"]

#         frames = contents["frames"]
#         for idx, frame in enumerate(frames):
#             cam_name = os.path.join(path, frame["file_path"] + extension)

#             # NeRF 'transform_matrix' is a camera-to-world transform
#             c2w = np.array(frame["transform_matrix"])
#             # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
#             c2w[:3, 1:3] *= -1

#             # get the world-to-camera transform and set R, T
#             w2c = np.linalg.inv(c2w)
#             R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
#             T = w2c[:3, 3]

#             image_path = os.path.join(path, cam_name)
#             if not os.path.isfile(image_path):
#                 image_path = os.path.join(path, "images", frame["file_path"] + extension)
#             image_name = Path(cam_name).stem
#             image = Image.open(image_path)

#             im_data = np.array(image.convert("RGBA"))

#             bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

#             norm_data = im_data / 255.0
#             arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
#             image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

#             fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
#             FovY = fovy 
#             FovX = fovx

#             cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, objects=None,
#                             image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
#     return cam_infos
def readCamerasFromTransforms(path, transformsfile, object_path,target_name,white_background,extension=".jpg"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        

        w = contents["w"]
        h = contents["h"]
        fl_x = contents["fl_x"]
        fl_y = contents["fl_y"]

        fovx = 2 * math.atan(w / (2 * fl_x))
        fovy = 2 * math.atan(h / (2 * fl_y))
        # ----------------------------

        frames = contents["frames"]
        for idx, frame in enumerate(frames):

            rel_path = frame["file_path"]
            image_path = os.path.join(path, rel_path + extension)
            

            
            if not os.path.isfile(image_path):
                image_path = os.path.join(path, "images", rel_path + extension)
            
            if not os.path.isfile(image_path):
                print(f"Warning: Could not find image at {image_path}")
                continue

            if target_name is not None and target_name != "":
                base_name = os.path.basename(rel_path)

                object_path1 = os.path.join(path,object_path,target_name)
                full_mask_path = os.path.join(object_path1, base_name + ".png")
                if os.path.exists(full_mask_path):
                    mask_image = Image.open(full_mask_path)
                    # Convert mask to tensor for contribution counting
                    objects = torch.from_numpy(np.array(mask_image)).to(torch.uint8)
                else:

                    
                    continue
            else:

                objects = torch.zeros((h, w), dtype=torch.uint8)
                

            image_name = Path(rel_path).stem
            image = Image.open(image_path)

            c2w_raw = np.array(frame["transform_matrix"])

            if not np.all(np.isfinite(c2w_raw)):
                print(f"warning: skipping frame with NaN/Inf: {frame['file_path']}")
                continue
            
            det = np.linalg.det(c2w_raw[:3, :3])
            if np.abs(det) < 1e-6:
                print(f"skipping corrupted frame: {rel_path} (singular matrix, det={det})")
                continue
            c2w = c2w_raw.copy()

            c2w[:3, 1:3] *= -1

            try:
                w2c = np.linalg.inv(c2w)
            except np.linalg.LinAlgError:
                print(f"skipping non-invertible frame: {rel_path}")
                continue

            # w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])
            T = w2c[:3, 3]

            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0

            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.uint8), "RGB")

            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, objects=objects,
                                image_path=image_path, image_name=image_name, width=w, height=h))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, object_path,target_name,extension=".jpg"):
    # print("Reading Training Transforms")

    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json",object_path,target_name, white_background,extension)
    # print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", object_path,target_name,white_background, extension)
    
    if len(train_cam_infos) == 0:
        print(f"CRITICAL WARNING: no valid mask images for category {target_name}")

        return None

    if not eval:
        print("No test split; test images merged into train")
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
    else:
        print("Using test split")

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        # num_pts = 100_000
        num_pts = 5000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}