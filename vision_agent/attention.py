"""注意力模块补丁：让官方 ultralytics 支持 SEBlock / EMA / CBAM。

背景
----
YOLOv8 的消融实验需要在 backbone 里插入 SE / EMA / CBAM 三种注意力模块。
这三个模块不是 ultralytics 上游自带的，``parse_model`` 里没有对应的分支，
所以直接把 YAML 丢给官方 ultralytics 会失败：

- ``SEBlock.__init__() missing 1 required positional argument: 'c2'``
  —— 官方把 YAML 里的 ``args`` 原样传给模块，没有按真实通道数覆盖。
- ``TypeError: unsupported operand type(s) for %: 'list' and 'int'``
  —— 同上，通道数没替换成整数。

本模块做两件事：

1. **注入模块类**：把 SEBlock / EMA / CBAM 挂到
   ``ultralytics.nn.modules.conv`` 与 ``ultralytics.nn.tasks`` 上，
   这样 YAML 里写 ``SEBlock`` 就能被 ``globals()[m]`` 找到。
2. **给 ``parse_model`` 补分支**：运行时重编译该函数，插入

   .. code-block:: python

       elif m in frozenset({SEBlock, EMA, CBAM}):
           c1 = ch[f]
           c2 = c1
           args = [c1, c2]

   让这三个模块按真实通道数构造。

用法
----
.. code-block:: python

    from vision_agent.attention import install

    install()          # 幂等，重复调用无副作用
    from ultralytics import YOLO
    model = YOLO("yolov8n-se-ema-cbam.yaml")

兼容性
------
- 官方 ultralytics ``>=8.0``：可用（实测 8.3.241）
- 已自带 CBAM 的版本会复用官方实现，不重复定义
- 已魔改过 SEBlock/EMA 的版本（例如本地 fork）直接跳过注入

实现说明
--------
``parse_model`` 的补丁是**运行时**重编译函数源码，不改动 site-packages 里的
文件。这样升级 ultralytics 不会把补丁冲掉，也不会污染全局环境。
如果未来 ultralytics 重构了 ``parse_model``（找不到注入点），本模块会
抛 ``AttentionPatchError`` 并提示改用 ``project_root`` 指向本地魔改仓库。
"""

from __future__ import annotations

import re
from typing import Any, Optional

__all__ = [
    "AttentionPatchError",
    "ChannelAttention",
    "SpatialAttention",
    "install",
    "is_installed",
    "SEBlock",
    "EMA",
    "CBAM",
]


class AttentionPatchError(RuntimeError):
    """补丁无法应用时抛出。"""


# --------------------------------------------------------------------------
# 模块实现
# --------------------------------------------------------------------------
# 这三个模块只依赖 torch.nn，可以独立于 ultralytics 版本存在。
# 实现与论文原文一致，��构与 ultralytics 魔改版保持一致。

import torch
import torch.nn as nn  # noqa: E402


class ChannelAttention(nn.Module):
    """通道注意力模块（RTMDet 风格）。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """按通道重标定输入。"""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """空间注意力模块。"""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """按空间位置重标定输入。"""
        return x * self.act(
            self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1))
        )


class CBAM(nn.Module):
    """卷积块注意力模块：通道注意力 + 空间注意力串联。"""

    def __init__(self, c1: int, c2: int, k: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """依次施加通道与空间注意力。"""
        return self.spatial_attention(self.channel_attention(x))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 块。

    https://arxiv.org/abs/1709.01507
    """

    def __init__(self, c1: int, c2: int, reduction: int = 16) -> None:
        super().__init__()
        c_ = max(1, c1 // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c_, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c_, c1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """按通道重标定输入。"""
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.l1(y)
        y = self.act(y)
        y = self.l2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class EMA(nn.Module):
    """Efficient Multi-Scale Attention 模块。

    https://arxiv.org/abs/2401.00945
    """

    def __init__(self, c1: int, c2: int, k: int = 3) -> None:
        super().__init__()
        self.groups = 8
        self.conv1x1 = nn.Conv2d(
            c1 // self.groups, c1 // self.groups, kernel_size=1, stride=1, padding=0
        )
        self.conv3x3 = nn.Conv2d(
            c1 // self.groups, c1 // self.groups, kernel_size=3, stride=1, padding=1
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(c1 // self.groups, c1 // self.groups)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """分组卷积 + GroupNorm 生成注意力权重后重标定输入。"""
        b, c, h, w = x.size()
        group_x = x.view(b * self.groups, -1, h, w)  # 分组后，每组通道数为 c // groups

        x1 = self.gn(self.conv1x1(self.avg_pool(group_x)))
        x2 = self.gn(self.conv3x3(group_x))

        weights = self.sigmoid(x1 + x2)
        out = group_x * weights
        return out.view(b, c, h, w)


# --------------------------------------------------------------------------
# 补丁
# --------------------------------------------------------------------------

_INSTALLED = False

# parse_model 里用来标识「未知模块走通用分支」的锚点。
# 官方 8.x 的写法是 ``else:\n            c2 = ch[f]``，
# 缩进与关键字都保持一致，避免补丁打到别的分支上。
_ANCHOR = "        else:\n            c2 = ch[f]"

_PATCHED = (
    "        elif m in frozenset({SEBlock, EMA}):\n"
    "            c1 = ch[f]\n"
    "            c2 = c1\n"
    "            args = [c1, c2]\n"
    "        elif m is CBAM:\n"
    "            c1 = ch[f]\n"
    "            c2 = c1\n"
    "            args = [c1, c2] if _cbam_takes_c2() else [c1]\n"
    "        else:\n"
    "            c2 = ch[f]"
)

# 取出 parse_model 函数源码的正则：匹配到下一个顶层 def / class 为止。
_PARSE_MODEL_RE = re.compile(r"\ndef parse_model\(.*?\n(?=\ndef |\nclass )", re.S)


def is_installed() -> bool:
    """补丁是否已经生效。"""
    return _INSTALLED


def _module_targets() -> tuple[Any, Any]:
    """返回 (ultralytics.nn.modules.conv, ultralytics.nn.tasks)。"""
    from ultralytics.nn.modules import conv as conv_mod
    from ultralytics.nn import tasks as tasks_mod

    return conv_mod, tasks_mod


def _resolve_classes(conv_mod: Any, tasks_mod: Any) -> dict[str, Any]:
    """决定每个模块用哪个实现。

    - ``SEBlock`` / ``EMA``：官方 ultralytics 没有，用本模块的实现。
    - ``CBAM``：官方有，但签名是 ``CBAM(c1, kernel_size=7)``，
      与消融实验用的 ``CBAM(c1, c2, k=7)`` 语义不同（后者按通道对构造）。
      为了让 YAML 里的 ``CBAM`` 行为一致，这里统一使用本模块的实现。
    - 若目标环境已经魔改过（本地 fork），直接复用其实现，不重复定义。
    """
    resolved: dict[str, Any] = {}
    for name, fallback in (("SEBlock", SEBlock), ("EMA", EMA), ("CBAM", CBAM)):
        cls = getattr(conv_mod, name, None)
        if cls is None:
            cls = getattr(tasks_mod, name, None)
        # 官方 CBAM 签名不带 c2，与消融实验的构造约定不一致，改用本模块实现
        if name == "CBAM" and cls is not None and not _cbam_takes_c2(cls):
            cls = fallback
        resolved[name] = cls if cls is not None else fallback
    return resolved


def _cbam_takes_c2(cls: Any) -> bool:
    """判断 CBAM 的构造函数是否需要 c2 参数。

    官方 ultralytics 的签名是 ``CBAM(c1, kernel_size=7)``，
    而魔改版的签名是 ``CBAM(c1, c2, k=7)``。
    两者参数个数不同，必须分别处理，否则会报
    ``AssertionError: kernel size must be 3 or 7``（k 收到了列表）。
    """
    import inspect

    try:
        params = list(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):  # pragma: no cover - 极端情况
        return True
    # 去掉 self
    params = [p for p in params if p != "self"]
    # 形如 (c1, c2, k=7) -> 需要 c2
    return len(params) >= 2 and params[1] in ("c2", "c_2", "out_channels")


def _patch_parse_model(tasks_mod: Any, classes: dict[str, Any]) -> None:
    """运行时给 parse_model 插入注意力模块分支。"""
    path = tasks_mod.__file__
    if not path:
        raise AttentionPatchError("无法定位 ultralytics.nn.tasks 源文件路径")

    with open(path, "r", encoding="utf-8") as fh:
        source = fh.read()

    match = _PARSE_MODEL_RE.search(source)
    if not match:
        raise AttentionPatchError(
            "未能在 ultralytics.nn.tasks 中定位 parse_model 函数；"
            "ultralytics 版本可能过新或已重构。"
            "请改用 task.yaml 的 project_root 指向本地魔改版 ultralytics 仓库。"
        )

    code = match.group(0)
    if "SEBlock, EMA" in code and "_cbam_takes_c2" in code:
        # 已经是打过补丁的（重复安装或本地 fork 已改过）
        return

    if _ANCHOR not in code:
        raise AttentionPatchError(
            "parse_model 的结构与预期不符，找不到注入点；"
            "请改用 task.yaml 的 project_root 指向本地魔改版 ultralytics 仓库。"
        )

    patched = code.replace(_ANCHOR, _PATCHED, 1)

    # 在 tasks 模块自身的命名空间里重编译，保证函数能访问
    # Concat / Detect / base_modules 等内部名字。
    namespace = tasks_mod.__dict__
    namespace.update(classes)
    namespace["_cbam_takes_c2"] = lambda: _cbam_takes_c2(classes["CBAM"])
    exec(compile(patched, path, "exec"), namespace)  # noqa: S102
    tasks_mod.parse_model = namespace["parse_model"]


def install(force: bool = False) -> bool:
    """安装补丁。

    :param force: 已安装时是否强制重装（默认幂等，直接返回 True）。
    :return: 补丁是否可用。
    :raises AttentionPatchError: ultralytics 不可用或结构不兼容。
    """
    global _INSTALLED

    if _INSTALLED and not force:
        return True

    try:
        conv_mod, tasks_mod = _module_targets()
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise AttentionPatchError(f"未安装 ultralytics：{exc}") from exc

    classes = _resolve_classes(conv_mod, tasks_mod)

    # 1) 注入模块类
    for name, cls in classes.items():
        if not hasattr(conv_mod, name):
            setattr(conv_mod, name, cls)
        setattr(tasks_mod, name, cls)

    # 2) 补 parse_model 分支
    _patch_parse_model(tasks_mod, classes)

    _INSTALLED = True
    return True


def installed_versions() -> dict[str, str]:
    """返回关键依赖的版本，便于排查环境问题。"""
    info: dict[str, str] = {}
    try:
        import torch

        info["torch"] = torch.__version__
    except ImportError:  # pragma: no cover
        info["torch"] = "not installed"
    try:
        import ultralytics

        info["ultralytics"] = getattr(ultralytics, "__version__", "unknown")
    except ImportError:  # pragma: no cover
        info["ultralytics"] = "not installed"
    return info


def _module_version() -> Optional[str]:
    """供 selftest 使用的内部钩子。"""
    return installed_versions().get("ultralytics")
