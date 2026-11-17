"""工作区规划。

Agent 的一切产物都落在一个独立的工作目录里，方便：
    - 多次实验之间互不污染；
    - 断点续跑（每一步的产物都在约定位置）；
    - 整体打包/迁移。

目录结构::

    <workspace>/
      task.yaml            任务配置快照
      state.json           流水线状态与产物清单
      README.md            本次运行说明
      research/            S1 调研产物
        raw/               各站点原始返回
        research_report.md
        datasets.json
        papers.json
        improvements.md
      models/              S2 克隆的官方模型仓库
      datasets/            S3/S4 数据集
        raw/               原始下载
        processed/         images|labels / train|val|test
        dataset_stats.json
        preprocess_report.md
      configs/             生成的 data.yaml 与模型 YAML
        models/
      scripts/             S5-S7 生成的脚本
        train_*.py
        infer_*.py
        app_*.py
      runs/                训练输出 + 消融汇总
        ablation_summary.md
        ablation_summary.csv
        best_run.json
      reports/             S7/S8 报告
        inference_report.md
        project_report.md
        figures/
      app/                 可视化系统运行时目录
        uploads/  outputs/
      logs/                运行日志
        agent.log
        train_<variant>.log
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workspace:
    """工作区路径集合。"""

    root: Path

    # --- 顶层 ---
    @property
    def task_file(self) -> Path:
        return self.root / "task.yaml"

    @property
    def state_file(self) -> Path:
        return self.root / "state.json"

    @property
    def readme(self) -> Path:
        return self.root / "README.md"

    # --- 分区 ---
    @property
    def research(self) -> Path:
        return self.root / "research"

    @property
    def research_raw(self) -> Path:
        return self.research / "raw"

    @property
    def research_report(self) -> Path:
        return self.research / "research_report.md"

    @property
    def research_datasets(self) -> Path:
        return self.research / "datasets.json"

    @property
    def research_papers(self) -> Path:
        return self.research / "papers.json"

    @property
    def research_improvements(self) -> Path:
        return self.research / "improvements.md"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def data_raw(self) -> Path:
        return self.datasets / "raw"

    @property
    def data_processed(self) -> Path:
        return self.datasets / "processed"

    @property
    def dataset_stats(self) -> Path:
        return self.datasets / "dataset_stats.json"

    @property
    def preprocess_report(self) -> Path:
        return self.datasets / "preprocess_report.md"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def model_configs(self) -> Path:
        return self.configs / "models"

    @property
    def data_yaml(self) -> Path:
        return self.configs / "data.yaml"

    @property
    def scripts(self) -> Path:
        return self.root / "scripts"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def ablation_md(self) -> Path:
        return self.runs / "ablation_summary.md"

    @property
    def ablation_csv(self) -> Path:
        return self.runs / "ablation_summary.csv"

    @property
    def best_run(self) -> Path:
        return self.runs / "best_run.json"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def inference_report(self) -> Path:
        return self.reports / "inference_report.md"

    @property
    def project_report(self) -> Path:
        return self.reports / "project_report.md"

    @property
    def app_usage(self) -> Path:
        return self.reports / "app_usage.md"

    @property
    def app_dir(self) -> Path:
        return self.root / "app"

    @property
    def app_uploads(self) -> Path:
        return self.app_dir / "uploads"

    @property
    def app_outputs(self) -> Path:
        return self.app_dir / "outputs"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    # --- 操作 ---
    def all_dirs(self) -> list[Path]:
        return [
            self.root,
            self.research,
            self.research_raw,
            self.models,
            self.datasets,
            self.data_raw,
            self.data_processed,
            self.configs,
            self.model_configs,
            self.scripts,
            self.runs,
            self.reports,
            self.figures,
            self.app_dir,
            self.app_uploads,
            self.app_outputs,
            self.logs,
        ]

    def ensure(self) -> "Workspace":
        """创建全部目录（幂等）。"""
        for d in self.all_dirs():
            d.mkdir(parents=True, exist_ok=True)
        return self

    def rel(self, path: Path) -> str:
        """相对工作区的可读路径，用于日志。"""
        try:
            return str(Path(path).resolve().relative_to(self.root.resolve()))
        except ValueError:
            return str(path)
