"""检索子包：数据集站点 / 学术文献 / GitHub 官方仓库。"""

from . import datasets, github, papers  # noqa: F401

__all__ = ["datasets", "papers", "github"]
