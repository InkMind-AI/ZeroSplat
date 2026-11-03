"""
ZeroSplat camera wrappers derived from Inria GRAPHDECO 3D Gaussian Splatting
(non-commercial research use). Extended with optional object masks for lifting.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from utils.graphics_utils import getProjectionMatrix, getWorld2View2

class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        gt_alpha_mask,
        image_name,
        uid,
        gt_depth,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
        objects=None,
        c2w=None,
    ):
        super().__init__()
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.c2w = c2w

        try:
            self.data_device = torch.device(data_device)
        except Exception as exc:
            print(exc)
            print(
                f"[Warning] device {data_device} unavailable; falling back to cuda"
            )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None and isinstance(gt_alpha_mask, np.ndarray):
            gt_alpha_mask = torch.from_numpy(gt_alpha_mask).to(self.data_device)
            gt_depth = torch.from_numpy(gt_depth).to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask
            self.gt_depth = gt_depth

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones(
                (1, self.image_height, self.image_width), device=self.data_device
            )

        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale))
            .transpose(0, 1)
            .cuda()
        )
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.objects = objects.to(self.data_device) if objects is not None else None

class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = torch.inverse(self.world_view_transform)[3, :3]
