"""环境自检：把「能不能训练」这件事变成一份可执行的检查报告。

要回答四个问题：
    1. 有没有装 Anaconda / Miniconda？（用户原话：如果环境没装，提醒用户自己装一个 anaconda）
    2. 有没有能跑 PyTorch 的 Python 解释器？装在哪？
    3. 这个解释器里有没有 torch / ultralytics / opencv 等依赖？
    4. 有没有可用的 GPU？

第 2 点最关键：Agent 自身可能跑在一个「没有 torch」的轻量 Python 里，
但训练必须用装了 torch 的解释器。所以这里提供 `resolve_training_python()`，
在当前解释器、conda 各环境、常见安装目录里挑出最合适的那个。
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

WINDOWS = os.name == "nt"

_CONDA_DIR_CANDIDATES = [
    "C:/ProgramData/anaconda3",
    "C:/ProgramData/miniconda3",
    "C:/anaconda3",
    "C:/miniconda3",
    str(Path.home() / "anaconda3"),
    str(Path.home() / "miniconda3"),
    str(Path.home() / ".conda"),
    "D:/anaconda3",
    "D:/miniconda3",
    "D:/ProgramData/anaconda3",
]

_REQUIRED_FOR_TRAIN = ["torch"]
_RECOMMENDED = ["torchvision", "ultralytics", "cv2", "yaml", "numpy", "pandas", "matplotlib"]
_REQUIRED_FOR_APP = ["flask", "cv2", "PIL"]


@dataclass
class EnvReport:
    conda_path: Optional[str] = None
    conda_version: str = ""
    conda_envs: List[str] = field(default_factory=list)
    python_exe: str = ""
    python_version: str = ""
    packages: Dict[str, str] = field(default_factory=dict)
    cuda_available: bool = False
    gpu_name: str = ""
    gpu_vram_mb: int = 0
    torch_version: str = ""
    ready_for_training: bool = False
    ready_for_app: bool = False
    instructions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "conda_path": self.conda_path,
            "conda_version": self.conda_version,
            "conda_envs": self.conda_envs,
            "python_exe": self.python_exe,
            "python_version": self.python_version,
            "packages": self.packages,
            "cuda_available": self.cuda_available,
            "gpu_name": self.gpu_name,
            "gpu_vram_mb": self.gpu_vram_mb,
            "torch_version": self.torch_version,
            "ready_for_training": self.ready_for_training,
            "ready_for_app": self.ready_for_app,
            "instructions": self.instructions,
        }

    def to_markdown(self) -> str:
        lines = ["# 环境自检报告", ""]
        lines.append("## 概览")
        lines.append("")
        lines.append("| 项目 | 结果 |")
        lines.append("| --- | --- |")
        lines.append(f"| Anaconda/Miniconda | {self.conda_path or '未检测到'} |")
        lines.append(f"| conda 版本 | {self.conda_version or '-'} |")
        lines.append(f"| conda 环境数 | {len(self.conda_envs)} |")
        lines.append(f"| 训练解释器 | {self.python_exe or '未找到'} |")
        lines.append(f"| Python 版本 | {self.python_version or '-'} |")
        lines.append(f"| PyTorch | {self.torch_version or '未安装'} |")
        lines.append(f"| CUDA 可用 | {'是' if self.cuda_available else '否'} |")
        lines.append(f"| GPU | {self.gpu_name or '-'} |")
        lines.append(f"| 显存 | {self.gpu_vram_mb} MB |")
        lines.append(f"| 可训练 | {'OK' if self.ready_for_training else 'NO'} |")
        lines.append(f"| 可运行可视化系统 | {'OK' if self.ready_for_app else 'NO'} |")
        lines.append("")
        if self.conda_envs:
            lines.append("## 发现的 conda 环境")
            lines.append("")
            for env in self.conda_envs:
                lines.append(f"- {env}")
            lines.append("")
        lines.append("## 需要执行的操作")
        lines.append("")
        if self.instructions:
            for item in self.instructions:
                lines.append(f"1. {item}")
        else:
            lines.append("无，环境已就绪。")
        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 基础探测
# ---------------------------------------------------------------------------


def _run(cmd: List[str], timeout: float = 20.0) -> str:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return (proc.stdout or "").strip()
    except Exception:
        return ""


def find_conda() -> Optional[Path]:
    """定位 conda 可执行文件：先看 PATH，再看常见安装目录。"""
    exe = "conda.exe" if WINDOWS else "conda"
    for env_var in ("CONDA_EXE", "CONDA_PREFIX"):
        val = os.environ.get(env_var)
        if val and Path(val).exists() and Path(val).name.lower().startswith("conda"):
            return Path(val)

    found = _run(["where", exe] if WINDOWS else ["which", exe])
    if found:
        first = found.splitlines()[0].strip()
        if first and Path(first).exists():
            return Path(first)

    for base in _CONDA_DIR_CANDIDATES:
        p = Path(base)
        if not p.exists():
            continue
        for rel in ("Scripts/conda.exe", "condabin/conda.bat", "bin/conda", "Scripts/conda"):
            cand = p / rel
            if cand.exists():
                return cand
    return None


def conda_version(conda_exe: Optional[Path]) -> str:
    if not conda_exe:
        return ""
    out = _run([str(conda_exe), "--version"])
    return out.splitlines()[0].strip() if out else ""


def list_conda_envs(conda_exe: Optional[Path]) -> List[str]:
    """列出 conda 环境（名称 + 路径）。"""
    envs: List[str] = []
    if conda_exe:
        out = _run([str(conda_exe), "env", "list"])
        for line in out.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                envs.append(f"{parts[0]}  ->  {parts[1]}")
    # 用户级环境目录
    user_envs = Path.home() / ".conda" / "envs"
    if user_envs.exists():
        for d in sorted(user_envs.iterdir()):
            if d.is_dir() and not any(str(d) in e for e in envs):
                envs.append(f"{d.name}  ->  {d}")
    return envs


def probe_python(python_exe: Path, packages: Optional[List[str]] = None) -> Dict[str, str]:
    """在目标解释器里检查依赖，返回 {包名: 版本 或 present 或 MISSING}。

    注意：模块名与发行包名经常不一致（cv2 -> opencv-python，PIL -> pillow，
    yaml -> pyyaml），所以用 importlib.util.find_spec 判断「能不能导入」，
    再用 importlib.metadata 去找对应发行包的版本。
    """
    packages = packages or (_REQUIRED_FOR_TRAIN + _RECOMMENDED + _REQUIRED_FOR_APP)
    code = (
        "import importlib.util as u, importlib.metadata as m, json, sys;"
        "alias={'cv2':['opencv-python','opencv-contrib-python','opencv-python-headless'],"
        "'PIL':['pillow'],'yaml':['pyyaml'],'sklearn':['scikit-learn'],"
        "'bs4':['beautifulsoup4'],'cv2':['opencv-python']};"
        f"names={packages!r};"
        "res={};\n"
        "for n in names:\n"
        "    try:\n"
        "        ok = u.find_spec(n) is not None\n"
        "    except Exception:\n"
        "        ok = False\n"
        "    ver=''\n"
        "    for cand in alias.get(n,[])+[n]:\n"
        "        try:\n"
        "            ver=m.version(cand); break\n"
        "        except Exception:\n"
        "            pass\n"
        "    res[n]= ver if ver else ('present' if ok else 'MISSING')\n"
        "print(json.dumps({'py':sys.version.split()[0],'pkgs':res}))"
    )
    try:
        proc = subprocess.run(
            [str(python_exe), "-c", code],
            capture_output=True,
            text=True,
            timeout=40,
            encoding="utf-8",
            errors="replace",
        )
        import json as _json

        data = _json.loads(proc.stdout.strip().splitlines()[-1])
        out = {"python_version": data.get("py", "")}
        out.update(data.get("pkgs", {}))
        return out
    except Exception:
        return {"python_version": "", **{p: "MISSING" for p in packages}}


def probe_torch_cuda(python_exe: Path) -> Dict[str, object]:
    code = (
        "import json;import torch as t;"
        "print(json.dumps({'ver':t.__version__,"
        "'cuda':bool(t.cuda.is_available()),"
        "'gpu':(t.cuda.get_device_name(0) if t.cuda.is_available() else ''),"
        "'cuda_ver':(t.version.cuda or '')}))"
    )
    try:
        proc = subprocess.run(
            [str(python_exe), "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            encoding="utf-8",
            errors="replace",
        )
        import json as _json

        return _json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {"ver": "", "cuda": False, "gpu": "", "cuda_ver": ""}


def candidate_pythons(conda_exe: Optional[Path]) -> List[Path]:
    """所有候选解释器：当前解释器 + conda 各环境 + 常见目录。"""
    cands: List[Path] = [Path(sys.executable)]
    if conda_exe:
        out = _run([str(conda_exe), "env", "list"])
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                exe = Path(parts[1]) / ("python.exe" if WINDOWS else "bin/python")
                if exe.exists():
                    cands.append(exe)
    user_envs = Path.home() / ".conda" / "envs"
    if user_envs.exists():
        for d in sorted(user_envs.iterdir()):
            exe = d / ("python.exe" if WINDOWS else "bin/python")
            if exe.exists():
                cands.append(exe)
    # 去重保序
    seen, uniq = set(), []
    for c in cands:
        key = str(c).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def resolve_training_python(prefer_cuda: bool = True) -> tuple[Optional[Path], Dict[str, str]]:
    """挑一个最合适的训练解释器。

    评分规则（分高者胜）：
        装了 torch            +100   （硬门槛）
        CUDA 可用             +40
        装了 ultralytics      +20
        装了 opencv(cv2)      +8
        装了 flask            +6
        就是当前解释器        +5
    没有装 torch 的解释器一律不选（返回 None）。
    """
    conda_exe = find_conda()
    best: Optional[Path] = None
    best_score = -1
    best_info: Dict[str, str] = {}
    for exe in candidate_pythons(conda_exe):
        info = probe_python(exe, ["torch", "ultralytics", "cv2", "flask"])
        score = 0
        if info.get("torch") and info.get("torch") != "MISSING":
            score += 100
            torch_info = probe_torch_cuda(exe)
            if torch_info.get("cuda"):
                score += 40
            if info.get("ultralytics") and info["ultralytics"] != "MISSING":
                score += 20
            if info.get("cv2") and info["cv2"] != "MISSING":
                score += 8
            if info.get("flask") and info["flask"] != "MISSING":
                score += 6
        if exe == Path(sys.executable):
            score += 5
        if score > best_score:
            best, best_score, best_info = exe, score, info
    if best_score < 100:  # 没有带 torch 的解释器
        return None, best_info
    return best, best_info


# ---------------------------------------------------------------------------
# 汇总报告
# ---------------------------------------------------------------------------


def query_gpu_vram_mb() -> int:
    """查询第一块 GPU 的显存（MB）。查不到返回 0。"""
    out = _run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], timeout=15
    )
    for line in out.splitlines():
        line = line.strip().replace("MiB", "").strip()
        if line.isdigit():
            return int(line)
    return 0


def suggest_batch(vram_mb: int, imgsz: int = 640, model: str = "yolov8n") -> int:
    """按显存估算一个相对稳妥的 batch size。

    经验基准：YOLOv8n @ 640 / batch 16 约占 3.5GB 显存（训练态）。
    再按输入尺寸平方缩放、按模型规模系数缩放，并只留 80% 显存给训练，
    最后向下取偶数，避免踩到碎片化显存。
    """
    if not vram_mb or vram_mb <= 0:
        return 0
    scale = "n"
    for candidate in ("n", "s", "m", "l", "x"):
        if str(model).lower().endswith(candidate):
            scale = candidate
            break
    model_factor = {"n": 1.0, "s": 1.45, "m": 2.3, "l": 3.6, "x": 5.2}[scale]
    per_batch_gb = 0.22 * model_factor * (max(64, imgsz) / 640.0) ** 2
    usable_gb = (vram_mb / 1024.0) * 0.8
    batch = int(usable_gb / per_batch_gb)
    batch = max(2, (batch // 2) * 2)
    return batch


def build_env_report(prefer_cuda: bool = True) -> EnvReport:
    report = EnvReport()
    conda_exe = find_conda()
    report.conda_path = str(conda_exe) if conda_exe else None
    report.conda_version = conda_version(conda_exe)
    report.conda_envs = list_conda_envs(conda_exe)

    python_exe, info = resolve_training_python(prefer_cuda=prefer_cuda)
    if python_exe:
        report.python_exe = str(python_exe)
        report.python_version = info.get("python_version", "")
        full = probe_python(python_exe, _REQUIRED_FOR_TRAIN + _RECOMMENDED + _REQUIRED_FOR_APP)
        report.packages = {k: v for k, v in full.items() if k != "python_version"}
        torch_info = probe_torch_cuda(python_exe)
        report.torch_version = str(torch_info.get("ver", ""))
        report.cuda_available = bool(torch_info.get("cuda"))
        report.gpu_name = str(torch_info.get("gpu", ""))
    else:
        report.python_exe = str(Path(sys.executable))
        report.python_version = sys.version.split()[0]
        report.packages = {p: "MISSING" for p in _REQUIRED_FOR_TRAIN + _RECOMMENDED}

    report.ready_for_training = bool(
        report.python_exe
        and report.packages.get("torch") not in (None, "MISSING")
        and report.packages.get("ultralytics") not in (None, "MISSING")
    )
    report.ready_for_app = bool(
        report.python_exe
        and report.packages.get("flask") not in (None, "MISSING")
        and report.packages.get("cv2") not in (None, "MISSING")
    )
    report.gpu_vram_mb = query_gpu_vram_mb()
    report.instructions = build_instructions(report)
    return report


def build_instructions(report: EnvReport) -> List[str]:
    """给出「缺什么就装什么」的可执行清单。"""
    steps: List[str] = []
    if not report.conda_path:
        steps.append(
            "未检测到 Anaconda/Miniconda。请先安装 Anaconda（https://www.anaconda.com/download），"
            "安装时勾选 “Add Anaconda to PATH”，然后重开终端。"
        )
    pkgs = report.packages or {}
    if pkgs.get("torch") in (None, "MISSING"):
        steps.append(
            "未检测到 PyTorch。在 Anaconda Prompt 中执行: "
            "conda create -n vision-agent python=3.10 -y && conda activate vision-agent && "
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        )
    if pkgs.get("ultralytics") in (None, "MISSING"):
        steps.append("未检测到 ultralytics。执行: pip install ultralytics（或 pip install -e <本地仓库>）")
    if pkgs.get("cv2") in (None, "MISSING"):
        steps.append("未检测到 opencv。执行: pip install opencv-python（摄像头视频流必需）")
    if pkgs.get("flask") in (None, "MISSING"):
        steps.append("未检测到 flask。执行: pip install flask（可视化系统必需）")
    if report.python_exe and not report.cuda_available and report.torch_version:
        steps.append("PyTorch 未检测到可用 GPU，训练将走 CPU（很慢）。建议安装 CUDA 版 PyTorch。")
    if not steps:
        steps.append("环境已就绪，无需额外安装。")
    return steps
