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

# Forward-only (inference) rasterizer: renders Gaussians and accumulates
# per-class accumulated weights. There is no backward pass.

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def rasterize_gaussians(
    mask_hw,
    label_ids,
    means3D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    if mask_hw is None:
        # mask_hw is indexed as mask_hw[y * W + x], i.e. an (H, W) map; all pixels -> class 1.
        mask_hw = torch.ones(raster_settings.image_height, raster_settings.image_width,
                             device=raster_settings.bg.device)
    mask_hw = mask_hw.float().contiguous()

    color, depth, alpha, radii, pixel_last_pos, class_weights, image_xys, gs_depth = _C.rasterize_gaussians(
        mask_hw,
        label_ids,
        raster_settings.bg,
        means3D,
        colors_precomp,
        opacities,
        scales,
        rotations,
        raster_settings.scale_modifier,
        cov3Ds_precomp,
        raster_settings.viewmatrix,
        raster_settings.projmatrix,
        raster_settings.tanfovx,
        raster_settings.tanfovy,
        raster_settings.image_height,
        raster_settings.image_width,
        sh,
        raster_settings.sh_degree,
        raster_settings.campos,
        raster_settings.prefiltered,
        raster_settings.num_classes,
        raster_settings.debug,
    )
    return color, radii, depth, alpha, pixel_last_pos, class_weights, image_xys, gs_depth

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool
    num_classes: int

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)

        return visible

    def forward(self, means3D, means2D, opacities, shs=None, colors_precomp=None, scales=None, rotations=None,
                cov3D_precomp=None, mask_hw=None, label_ids=None):
        # means2D is unused (no gradients are propagated); kept for API compatibility.

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide exactly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])
        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if label_ids is None:
            label_ids = torch.tensor([], dtype=torch.int)

        return rasterize_gaussians(
            mask_hw,
            label_ids,
            means3D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            self.raster_settings,
        )
