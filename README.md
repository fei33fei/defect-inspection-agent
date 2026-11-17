# defect-inspection-agent

工业表面缺陷检测的自动化实验流水线。你只需要写一份 `task.yaml`（数据集类型 + 选用模型 + 工作目录），它会自己完成：

查数据集与论文 → 取官方模型 → 下载数据 → 转标注 → 跑消融训练 → 搭可视化系统 → 做推理测试 → 写项目文档。

它是一个**独立的 Python 包**，不依赖任何 Agent 运行时。你在装了 Python 的机器上直接跑，产物落在工作目录里，可以直接进 Git、可以打包、可以交给别人复现。

```
vision_agent/
├── vision_agent/          # Agent 本体（可 pip install -e .，也可直接 python -m vision_agent）
│   └── attention.py       # 让官方 ultralytics 支持 SE/EMA/CBAM 的补丁
├── config/
│   └── task.example.yaml  # 任务配置模板
├── tests/
│   ├── fixtures/task.smoke.yaml   # 2 epoch 全链路冒烟配置
│   └── test_smoke.py
├── AGENT_DESIGN.md        # 设计说明
└── pyproject.toml
```

---

## 快速开始

```bash
git clone https://github.com/fei33fei/defect-inspection-agent.git
cd defect-inspection-agent

# 0. 环境体检（找 conda / torch / GPU，缺什么会列出来）
python -m vision_agent doctor

# 1. 复制一份配置并按需修改
cp config/task.example.yaml task.yaml

# 2. 只做计划推演，不执行任何阶段（建议先跑这步）
python -m vision_agent plan --task task.yaml

# 3. 正式执行
python -m vision_agent run --task task.yaml --dry-run   # 先看将要做什么
python -m vision_agent run --task task.yaml

# 4. 看进度
python -m vision_agent status --task task.yaml

# 5. 启动可视化系统（上传检测 + 摄像头视频流）
python -m vision_agent serve --task task.yaml
```

跑完 `serve` 后打开 <http://127.0.0.1:8000/>。

> **Agent 本体只依赖 `pyyaml` 和 `requests`**，用哪个 Python 跑都行。
> **训练**会自动挑一个装了 torch 的解释器（见下文「环境自检」）。

也可以装成命令行工具：

```bash
pip install -e .          # 之后直接用 vision-agent <命令>
pip install -e ".[app]"   # 额外装 flask / opencv-python，用于可视化系统
```

---

## 命令一览

| 命令 | 作用 |
| --- | --- |
| `init --task task.yaml` | 生成带注释的任务配置模板 |
| `doctor` | 检查 conda 环境 / PyTorch / GPU / 依赖完整度 |
| `self-test` | Agent 自检（15 项，不需要 pytest） |
| `plan --task task.yaml` | 推演全流程：阶段清单、消融矩阵、数据格式、环境就绪情况 |
| `run --task task.yaml` | 执行流水线（`--stages s4 s5` 指定阶段，`--resume` 断点续跑，`--dry-run` 空跑） |
| `status --task task.yaml` | 查看每个阶段的状态与产物 |
| `search --keywords "..."` | 单独检索数据集；加 `--papers` 检索论文 |
| `serve --task task.yaml` | 启动工作区里生成的可视化系统 |
| `report --task task.yaml` | 只重新生成项目文档 |

`run` 的常用参数：

| 参数 | 说明 |
| --- | --- |
| `--stages s4 s5` | 只跑指定阶段（支持 `s4-s7` 区间写法） |
| `--resume` | 跳过已完成阶段，从断点继续 |
| `--dry-run` | 打印将要执行的动作但不真跑（训练尤其建议先 dry-run） |
| `--no-train` | 跳过实际训练（只出脚本与汇总框架） |
| `--force-train` | 即使 `auto_train=false` 也训练 |
| `--reuse-runs` | 若 run 目录已有 `results.csv` 就直接复用（只重解析指标，不重训） |
| `--echo-train` | 训练日志实时打到终端 |
| `--split test` | 推理测试用哪个划分（划分不存在时自动回退并在报告标注） |
| `--keep-going` | 某阶段失败后继续执行后续阶段 |

---

## 九个阶段

每个阶段只认「约定路径上的文件」，因此可以断点续跑，产物也能直接进 Git 交付。

| 阶段 | 做什么 | 产物 |
| --- | --- | --- |
| `s0` | 建工作区、落配置快照 | `task.yaml` 快照、目录树、运行说明 |
| `s1` | 搜数据集与论文、归纳改进方向 | `research/research_report.md`、`datasets.json`、`papers.json`、`improvements.md` |
| `s2` | 取官方模型 | `models/`、`configs/models/`（自动生成的组合变体） |
| `s3` | 下载数据集 | `datasets/raw/`、`inventory.json`（含 SHA256）、`source_report.md` |
| `s4` | 预处理：检测框转 YOLO 坐标并校验 | `datasets/processed/`、`configs/data.yaml`、`preprocess_report.md` |
| `s5` | 写训练脚本并自动跑消融 | `scripts/train_*.py`、`runs/<实验>/`、`ablation_summary.md/.csv`、`best_run.json` |
| `s6` | 生成可视化系统 | `scripts/app_*.py`（Flask）、`reports/app_usage.md` |
| `s7` | 用最优权重做推理测试 | `reports/inference/metrics.json`、`inference_report.md`、标注样例图 |
| `s8` | 撰写项目文档 | `reports/project_report.md`（背景/数据来源/算法/训练/结果/推理/用法/总结） |

摄像头环节**由使用者本人操作**：Agent 只生成代码与说明，不默认占用摄像头和端口。

---

## 注意力模块补丁

SE / EMA / CBAM 这三个模块不是 ultralytics 上游自带的，`parse_model` 里没有对应分支，直接把 YAML 丢给官方 ultralytics 会失败：

```
SEBlock.__init__() missing 1 required positional argument: 'c2'
TypeError: unsupported operand type(s) for %: 'list' and 'int'
```

`vision_agent/attention.py` 解决这件事，做两件事：

1. **注入模块类** —— 把 SEBlock / EMA / CBAM 挂到 `ultralytics.nn.modules.conv` 与 `ultralytics.nn.tasks`；
2. **给 `parse_model` 补分支** —— 运行时重编译该函数，让这三个模块按真实通道数构造。

补丁是**运行时**生效的，不改动 site-packages 里的文件，升级 ultralytics 不会冲掉它。

```python
from vision_agent.attention import install

install()          # 幂等
from ultralytics import YOLO

model = YOLO("configs/models/yolov8n-se-ema-cbam.yaml")
```

已在 **ultralytics 8.3.241 + torch 2.8.0** 上验证：模型可构建（3.04M 参数）并正常前向。

如果你有一份自己魔改过的 ultralytics，设环境变量 `VA_PROJECT_ROOT` 指向它即可，补丁会自动跳过重复注入：

```bash
set VA_PROJECT_ROOT=D:/path/to/ultralytics
```

---

## 坐标格式校验

`s4` 会逐行校验标签：坐标是否在 `[0,1]`、框是否越界、类别索引是否越界、图签是否一一对应。

坐标问题是 YOLO 检测最常见的「训完发现 mAP 上不去」原因，这里在训练前就拦住。

---

## 环境自检

`doctor` 会输出：conda 是否安装、有哪些 conda 环境、哪个解释器装了 torch、CUDA 是否可用、显存多少，以及**缺什么就装什么**的命令清单。若找不到能跑 torch 的解释器，`s5` 会中止并把安装步骤写进 `logs/env_report.md`。

Agent 自身可以跑在没有 torch 的轻量 Python 里，训练/推理统一用 subprocess 调「装了 torch 的那个解释器」，两者互不污染。

### 显存自适应与 OOM 自愈

- `train.batch` 填 `0` 表示**按显存自动估算**（按输入尺寸与模型规模折算）；
- 填了偏大的 batch 时会先警告，再由 OOM 自愈兜底；
- 训练日志出现 `CUDA out of memory` 时，Agent 会**自动把 batch 减半并重跑**（最多 2 次），调整过程写进消融汇总表的「读数须知」，不会悄悄改数。

---

## 大模型接入

`s1`（调研归纳）与 `s8`（文档润色）需要大模型。凭据解析顺序：

1. `task.yaml` 的 `llm.base_url` / `llm.api_key`
2. 环境变量 `VISION_AGENT_LLM_API_KEY`（或 `llm.api_key_env` 指定的名字）
3. `~/.minimax/config.yaml` 里的 `provider.minimax`

协议自动适配 `anthropic`（`x-api-key` + `/messages`）与 `chat-completions`（Bearer + `/chat/completions`）。

**没有可用模型时会自动降级为确定性模板**，模板质量按「够用」来设计：

| 能力 | 模板模式下的实现 |
| --- | --- |
| 论文要点 | 从摘要里按关键词命中挑出最相关的句子 |
| 数据集规模 | 从描述里正则抽出图片数 / 类别数 / 标注格式 / 分辨率 |
| 改进方向 | **扫描本地 ultralytics 仓库里已实现的注意力模块**，列出可做消融的候选 |
| 诚实标注 | 文件里明确写「由确定性模板生成（未调用大模型润色）」，不假装调用过 |

---

## 工作区结构

所有产物都落在 `workspace` 配置的目录下：

```
<workspace>/
├── task.yaml          配置快照
├── state.json         阶段状态与产物清单
├── research/          s1 调研产物
├── models/            s2 克隆的官方仓库
├── datasets/          s3/s4 原始数据与转换后的数据集
├── configs/           data.yaml 与自动生成的模型 YAML
├── scripts/           train_*.py / infer_*.py / app_*.py
├── runs/              训练输出 + 消融汇总 + best_run.json
├── reports/           推理报告 + 项目文档 + 样例图
├── app/               可视化系统的上传与输出目录
└── logs/              agent.log、train_<实验>.log、env_report.md
```

---

## 可视化系统

`python -m vision_agent serve --task task.yaml` 后打开 <http://127.0.0.1:8000/>：

| 页面 / 接口 | 功能 |
| --- | --- |
| `/` | 上传本地图片 → 检测 → 预览标注图 → 下载；**可多选批量检测并生成对比图** |
| `/camera` | 摄像头实时检测（MJPEG 视频流），可切换摄像头序号或选视频文件 |
| `/api/predict` | POST 字段 `image`，返回 JSON（类别 / 置信度 / 坐标 / 耗时） |
| `/api/predict_batch` | POST 字段 `images`（可多文件），返回逐张结果 + 对比图链接 |
| `/api/cameras` | 探测可用摄像头序号 |
| `/health` | 模型加载状态 |

批量检测的对比图会把多张标注图拼成网格，每张带「文件名 + 类别 + 耗时」的标题条。

外接 USB 摄像头一般是 `1` 号及之后；`0` 号通常是内置摄像头。需要 `flask` 与 `opencv-python`（缺 opencv 时标注图保存会自动退回 PIL）。

---

## 依赖

Agent 本体：`pyyaml`、`requests`（可选 `flask` / `opencv-python` 用于可视化系统）。

训练环境（Agent 会自动挑一个满足条件的）：`torch`、`torchvision`、`ultralytics`、`opencv-python`。

```bash
pip install pyyaml requests

# 训练环境
conda create -n vision-agent python=3.10 -y
conda activate vision-agent
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python flask
```

---

## 冒烟测试

```bash
python -m vision_agent self-test
```

15 项自检，不需要 pytest，一键验证配置解析、坐标校验、代码生成、可视化接口等。

```bash
python -m vision_agent run --task tests/fixtures/task.smoke.yaml
```

用真实数据 + 真实 GPU 训练，但只跑 2 个 epoch、只跑基线，用来验证 s0→s8 全链路是否通。

---

## 常见问题

**Q：训练报 `SEBlock.__init__() missing 1 required positional argument: 'c2'`**
A：注意力模块补丁没生效。确认 `from vision_agent.attention import install; install()` 在 `from ultralytics import YOLO` **之前**执行。Agent 生成的脚本已经内置了这一步，手写脚本时需要自己加。

**Q：`doctor` 说找不到能跑 torch 的解释器**
A：`doctor` 的输出里会列出所有 conda 环境和它们各自装了 torch 的情况。装好之后重跑 `doctor` 即可。

**Q：`run` 跑到 s5 卡住不动**
A：先加 `--dry-run` 看它打算执行什么，再用 `--stages s4 s5` 单独跑训练阶段，加 `--echo-train` 看实时日志。

**Q：训练日志里出现 `CUDA out of memory`**
A：Agent 会自动把 batch 减半并重跑（最多 2 次）。想手动控制就把 `train.batch` 填小些，填 `0` 表示按显存自动估算。

**Q：改了配置想重新跑某一步**
A：`--resume` 会跳过已完成的阶段；`--stages sN` 只跑指定阶段。`--force-train` 可以让 `auto_train=false` 的配置也强制训练。

**Q：Windows 终端中文乱码**
A：PowerShell 5.1 的控制台编码不是 UTF-8。执行 `chcp 65001`，或改用 Windows Terminal。

**Q：数据下载失败 / 站点访问不稳定**
A：`s1` 的数据集检索依赖外网，站点不可达时会退化为「只有论文、无数据集候选」，不会阻塞后续流程。Kaggle 需要账号凭据，Agent 不代持凭据，只生成标准命令让你手动执行。

---

## 隐私

本项目不采集任何用户数据，不上传任何图片或权重到第三方。

- 可视化系统的上传图片只落在你自己机器的 `app/uploads/` 与 `app/outputs/` 目录；
- `s1` 的检索会访问 HuggingFace / Zenodo / arXiv / Semantic Scholar / Crossref / OpenAlex / GitHub 这些公开站点，**只发查询关键词，不发你的图片或数据**；
- 配置大模型后，`s1` 与 `s8` 会把调研文本与文档草稿发给你指定的模型端点——这是你自己配的，凭据来自 `task.yaml`、环境变量或 `~/.minimax/config.yaml`；
- 摄像头画面只在本地处理，不外发。

---

## License

本项目基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 深度定制，遵循 **AGPL-3.0** 协议，版权归原作者所有。

补丁部分（`vision_agent/attention.py`）中的 SEBlock / EMA / CBAM 实现同样出自本项目，随 AGPL-3.0 一并分发。

**若使用本项目，请引用上游工作。**
