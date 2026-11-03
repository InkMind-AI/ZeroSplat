"""Pose-alignment sanity check: render-vs-GT PSNR during the lifting pass.

The lifting pass already renders every training view with the trained Gaussians. If the
COLMAP cameras in ``<scene>/sparse/0`` are not the ones the Gaussians were trained with
(e.g. gr-lerf vs lerf COLMAP runs), those renders do not match the photos and the mask lift
lands on the wrong Gaussians.

Reference values measured on this project:
  * aligned (lerf figurines, lerf sparse):        PSNR ~21-30 dB
  * misaligned (figurines with gr-lerf sparse):   PSNR ~9-13 dB
  * slightly misaligned (grlerf ramen/teatime):   PSNR ~15-19 dB

Usage (see render.py):
    checker = PoseAlignmentChecker(tag=target_name)
    checker.update(rendering, gt)   # per view, [3,H,W] tensors in [0,1]
    checker.report()                # prints summary, warns (or raises) if misaligned

Env overrides:
    POSE_CHECK_MIN_PSNR   median-PSNR warning threshold in dB (default 18)
    POSE_CHECK_STRICT=1   raise RuntimeError instead of only warning
"""

import os

import torch

class PoseAlignmentChecker:
    def __init__(self, tag="", min_psnr=None, strict=None):
        self.tag = tag
        self.min_psnr = float(os.environ.get("POSE_CHECK_MIN_PSNR", 18.0)) if min_psnr is None else float(min_psnr)
        self.strict = (os.environ.get("POSE_CHECK_STRICT", "0") == "1") if strict is None else bool(strict)
        self._psnr = []  # list of 0-dim CUDA/CPU tensors; synced once in report()

    @torch.no_grad()
    def update(self, rendering, gt):
        gt = gt.to(rendering.device)
        if rendering.shape != gt.shape:
            return
        mse = ((rendering.clamp(0, 1) - gt.clamp(0, 1)) ** 2).mean().clamp_min(1e-10)
        self._psnr.append(-10.0 * torch.log10(mse))

    def report(self):
        tag = f"[{self.tag}] " if self.tag else ""
        if not self._psnr:
            print(f"[pose-check] {tag}skipped: no views were compared")
            return None

        psnr = torch.stack(self._psnr).float().cpu()
        median = psnr.median().item()
        mean = psnr.mean().item()
        frac_bad = (psnr < self.min_psnr).float().mean().item()
        print(f"[pose-check] {tag}render-vs-GT PSNR over {len(psnr)} views: "
              f"median={median:.1f} dB, mean={mean:.1f} dB, "
              f"min={psnr.min().item():.1f} dB, max={psnr.max().item():.1f} dB, "
              f"frac(<{self.min_psnr:g} dB)={frac_bad:.2f}")

        ok = median >= self.min_psnr
        if not ok:
            bar = "!" * 78
            msg = (
                f"{bar}\n"
                f"[pose-check] {tag}WARNING: median PSNR {median:.1f} dB < {self.min_psnr:g} dB.\n"
                f"  COLMAP cameras (<scene>/sparse/0) probably do NOT match the trained Gaussians\n"
                f"  (e.g. Gaussians trained on data/lerf/<scene>/sparse, but sparse/0 is from gr-lerf).\n"
                f"  2D masks will be lifted onto the wrong Gaussians; fix sparse/0 before trusting results.\n"
                f"{bar}"
            )
            print(msg)
            if self.strict:
                raise RuntimeError(f"pose-check failed for '{self.tag}': median PSNR {median:.1f} dB")
        return {"median": median, "mean": mean, "ok": ok}
