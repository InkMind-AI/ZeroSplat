# Zero Rasterization (ZeroSplat)

This module is **derived from the Inria GRAPHDECO 3D Gaussian Splatting (3DGS) rasterizer**
([diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization);
non-commercial research license, see `LICENSE.md`).

It provides a forward-only, multi-class segmentation rasterizer used by ZeroSplat.
If you use it in research, please cite 3D Gaussian Splatting (BibTeX below).

## Modifications relative to diff-gaussian-rasterization

- Forward-only (inference): removed backward CUDA kernels, the autograd wrapper, and training-time gradient paths.
- `rasterize_gaussians` no longer returns `num_rendered` and the geometry / binning / image buffers.
- Added multi-class segmentation outputs (`mask_hw`, `label_ids`, `class_weights`, `pixel_last_pos`, etc.).
- Identifiers renamed for the ZeroSplat Python/C++/CUDA API:

  | diff-gaussian-rasterization (typical) | zero-rasterization |
  |---------------------------------------|--------------------|
  | (N/A — segmentation extension)        | `mask_hw`          |
  |                                       | `label_ids`        |
  |                                       | `class_weights`    |
  |                                       | `pixel_last_pos`   |
  |                                       | `num_classes`      |
  |                                       | `class_id`         |

  The same names are used in CUDA kernels, C++ bindings, and the Python API.
  `projected_xy` in older wrappers is `image_xys` here, matching the C++/CUDA side.

## Acknowledgements

Built on open-source 3D Gaussian Splatting and related rasterization work. Thanks to the original authors.

- [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [diff-gaussian-rasterization-kiui](https://github.com/ashawkey/diff-gaussian-rasterization)

## BibTeX (3D Gaussian Splatting)

```bibtex
@Article{kerbl3Dgaussians,
  author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal      = {ACM Transactions on Graphics},
  number       = {4},
  volume       = {42},
  month        = {July},
  year         = {2023},
  url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```
