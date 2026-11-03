# ZeroSplat

**ZeroSplat: Generalized Referring Segmentation in 3D Gaussian Splatting**

Project page: https://inkmind-ai.github.io/ZeroSplat/

Benchmarks: [GR-LERF](https://huggingface.co/datasets/jiayuttkx/GR-LERF) · [GR-ScanNet](https://huggingface.co/datasets/jiayuttkx/GR-ScanNet) · [Ref-LERF](https://huggingface.co/datasets/FudanCVL/Ref-Lerf) · [LERF](https://www.lerf.io/) · [ScanNet](http://www.scan-net.org/)

This repository contains the **implementation** (pipeline + evaluation) and the project website (`index.html`, `static/` for GitHub Pages). Run everything via Python and the commands below.

## Repository layout

| Path | Role |
|------|------|
| `main.py` | End-to-end pipeline: voxel → VLM → SAM3 → render |
| `voxel_grid_main.py`, `mask_generator.py`, `render.py` | Individual stages (called by `main.py`) |
| `eval/` | Dataset evaluation entry points |
| `submodules/` | CUDA extensions (`simple-knn`, `diff-gaussian-rasterization`, `zero-rasterization`) |

Data, model weights, and SAM3 are **not** shipped. See [`.gitignore`](.gitignore).

## Installation

1. Create a Python environment with **PyTorch + CUDA** matching your driver.

2. Install submodule extensions (from repo root):

```bash
pip install -e submodules/simple-knn
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/zero-rasterization
pip install -r requirements.txt
```

3. Install **SAM3** at the repo root as `sam3_main/` (clone or symlink the official SAM3 repo). Place the checkpoint at `sam3_main/sam3.pt`. The pipeline imports `sam3_main.sam3`; a site-packages-only install is not enough.

4. **VLM**: use any OpenAI-compatible HTTP API (cloud API or a local server). [vLLM](https://github.com/vllm-project/vllm) is **not** in `requirements.txt`; install it in a separate environment (it is heavy and often conflicts with the pipeline env).

```bash
# Install vLLM (follow https://docs.vllm.ai if this wheel does not match your CUDA)
pip install vllm

# Download the vision-language weights (example used in the paper)
huggingface-cli download Qwen/Qwen3-VL-30B-A3B-Instruct \
  --local-dir /path/to/Qwen3-VL-30B-A3B-Instruct
```

Then start the server:

```bash
vllm serve /path/to/Qwen3-VL-30B-A3B-Instruct \
  --host 0.0.0.0 --port 8001 \
  --dtype bfloat16 --trust-remote-code \
  --served-model-name qwen3-vl-30b-a3b-instruct
```

Check readiness: `curl -s http://127.0.0.1:8001/v1/models`

## Data layout

Set `DATA` to the directory that contains dataset folders, e.g.:

```text
$DATA/
  grlerf/<scene>/...
  Ref-lerf/<scene>/...
  gr-scannet/<scene>/...
  scannet/<scene>/...
  lerf/<scene>/...
```

CLI `--dataset_type gr_scannet` maps to the on-disk folder `gr-scannet/`. Other types use the same name on disk (`grlerf`, `Ref-lerf`, `lerf`, `scannet`).

Each scene needs a reconstructed 3DGS model (`point_cloud/iteration_*/point_cloud.ply`), images (`images/` or ScanNet `color/`), and the corresponding GT / JSON used by that benchmark.

## Pipeline (required arguments)

All paths and VLM settings must be passed explicitly (no repo defaults for secrets or data roots).

```bash
export PY=/path/to/venv/bin/python
export DATA=/path/to/data
export CODE=/path/to/zerosplat
cd "$CODE"

$PY main.py \
  --base_data_dir "$DATA" \
  --gs_python "$PY" \
  --dataset_type Ref-lerf \
  --dataset_name ramen \
  --api_key EMPTY \
  --base_url http://127.0.0.1:8001/v1 \
  --model_name qwen3-vl-30b-a3b-instruct \
  --k 40 --threshold 0.8
```

- **`--dataset_type`**: `grlerf`, `Ref-lerf`, `lerf`, `gr_scannet`, or `scannet`
- **`--api_key`**: use literal `EMPTY` for local vLLM without auth; otherwise your API key
- **`--iteration`**: optional; default 30000 (LERF/GR-LERF) or 90000 (ScanNet / GR-ScanNet) inside `main.py`
- Stage skips: `--skip_voxel`, `--skip_mask`, `--skip_render`, `--proj_only`

GR-ScanNet example:

```bash
$PY main.py \
  --base_data_dir "$DATA" \
  --gs_python "$PY" \
  --dataset_type gr_scannet \
  --dataset_name scene0000_00 \
  --iteration 90000 \
  --api_key "$VLM_API_KEY" \
  --base_url "$VLM_BASE_URL" \
  --model_name "$VLM_MODEL_NAME" \
  --k 40 --threshold 0.8
```

## Evaluation

Run after predictions exist under each scene (e.g. `zero_out/`, `knn_label/`).

**Ref-LERF / GR-LERF** (`eval/lerf_reflerf/run_reflerf_eval.py`):

```bash
# GR-LERF
$PY eval/lerf_reflerf/run_reflerf_eval.py \
  --dataset_type grlerf \
  --scene_name ramen \
  --gt_base "$DATA/grlerf/ramen/mask" \
  --pred_base "$DATA/grlerf/ramen/zero_out" \
  --json_path "$DATA/grlerf/ramen/json" \
  --iteration 30000

# Ref-LERF (JSON lives under json/test_json)
$PY eval/lerf_reflerf/run_reflerf_eval.py \
  --dataset_type Ref-lerf \
  --scene_name ramen \
  --gt_base "$DATA/Ref-lerf/ramen/mask" \
  --pred_base "$DATA/Ref-lerf/ramen/zero_out" \
  --json_path "$DATA/Ref-lerf/ramen/json/test_json" \
  --iteration 30000
```

**LERF** (`eval/lerf_reflerf/run_lerf_eval.py`):

```bash
$PY eval/lerf_reflerf/run_lerf_eval.py \
  --scene_name waldo_kitchen \
  --gt_base "$DATA/lerf/label/waldo_kitchen/gt" \
  --pred_base "$DATA/lerf/waldo_kitchen/zero_out" \
  --iteration 30000
```

**GR-ScanNet** (`eval/gr_scannet/run_gr_scannet_eval.py`):

```bash
$PY eval/gr_scannet/run_gr_scannet_eval.py \
  --data_root "$DATA/gr-scannet" \
  --scene_name scene0000_00 \
  --iteration 90000 --k 40 --threshold 0.8
```

**ScanNet** (`eval/scannet/run_scannet_eval.py`):

```bash
$PY eval/scannet/run_scannet_eval.py \
  --data_root "$DATA/scannet" \
  --iteration 90000 --num_classes 19
# After render (KNN on by default), add: --use_knn --k 40 --threshold 0.8
```

### Batch GR-ScanNet (shell loop)

Replace scene names with your list:

```bash
for scene in scene0000_00 scene0062_00 scene0070_00; do
  $PY main.py --base_data_dir "$DATA" --gs_python "$PY" \
    --dataset_type gr_scannet --dataset_name "$scene" --iteration 90000 \
    --api_key "$VLM_API_KEY" --base_url "$VLM_BASE_URL" --model_name "$VLM_MODEL_NAME" \
    --k 40 --threshold 0.8
  $PY eval/gr_scannet/run_gr_scannet_eval.py \
    --data_root "$DATA/gr-scannet" --scene_name "$scene" \
    --iteration 90000 --k 40 --threshold 0.8
done
```

## Run stages separately

```bash
# Voxel / keyframes only
$PY voxel_grid_main.py --base_data_dir "$DATA" --dataset_type grlerf --dataset_name ramen --iteration 30000

# VLM + SAM3
$PY mask_generator.py --base_data_dir "$DATA" --dataset_type grlerf --dataset_name ramen \
  --api_key "$VLM_API_KEY" --base_url "$VLM_BASE_URL" --model_name "$VLM_MODEL_NAME"

# Render / lift labels
$PY render.py --base_data_dir "$DATA" --dataset_type grlerf --dataset_name ramen --k 40 --threshold 0.8
```

If local VLM requests fail unexpectedly, unset proxy variables (`ALL_PROXY`, `http_proxy`, `https_proxy`, and uppercase variants).

## License

See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Submodule licenses are under each `submodules/*/LICENSE*`.
