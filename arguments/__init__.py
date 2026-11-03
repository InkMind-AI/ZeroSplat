"""CLI / namespace helpers for ZeroSplat (3DGS ParamGroup pattern, localized)."""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser, Namespace


class GroupParams:
    """Bag bag for a subset of argparse values."""


class ParamGroup:
    """Register dataclass-like attributes on an ArgumentParser group."""

    def __init__(self, parser: ArgumentParser, name: str, fill_none: bool = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = key.startswith("_")
            arg_name = key[1:] if shorthand else key
            value = None if fill_none else value
            typ = type(value)
            if shorthand:
                flag = ("--" + arg_name, "-" + arg_name[:1])
                if typ is bool:
                    group.add_argument(*flag, default=value, action="store_true")
                else:
                    # sentinel 模式下 value 为 None，type 不能传 NoneType
                    group.add_argument(
                        *flag,
                        default=value,
                        type=None if typ is type(None) else typ,
                    )
            else:
                if typ is bool:
                    group.add_argument("--" + arg_name, default=value, action="store_true")
                else:
                    # sentinel 模式下 value 为 None，type 不能传 NoneType，
                    # 否则 set_defaults 的字符串默认值会被 NoneType() 转换报错
                    group.add_argument(
                        "--" + arg_name,
                        default=value,
                        type=None if typ is type(None) else typ,
                    )

    def extract(self, args: Namespace) -> GroupParams:
        group = GroupParams()
        self_vars = vars(self)
        for name, value in vars(args).items():
            if name in self_vars or ("_" + name) in self_vars:
                setattr(group, name, value)
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel: bool = False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = True
        # ZeroSplat stores SAM3 masks under sam3_mask by default
        self._object_path = "sam3_mask"
        self.target_name = ""
        self.test_image_names = []
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args: Namespace) -> GroupParams:
        group = super().extract(args)
        if getattr(group, "source_path", None):
            group.source_path = os.path.abspath(group.source_path)
        return group


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    """Retained for 3DGS scene loading compatibility; ZeroSplat itself is training-free."""

    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser) -> Namespace:
    """Merge CLI args with an optional saved cfg_args next to the model path."""
    cmdline = parser.parse_args(sys.argv[1:])
    cfg_path = None
    try:
        if cmdline.model_path:
            cfg_path = os.path.join(cmdline.model_path, "cfg_args")
    except Exception:
        cfg_path = None

    merged = Namespace()
    if cfg_path and os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as handle:
            cfg_ns = eval(handle.read())
        for key, value in vars(cfg_ns).items():
            setattr(merged, key, value)
    for key, value in vars(cmdline).items():
        if value is not None:
            setattr(merged, key, value)
    return merged
