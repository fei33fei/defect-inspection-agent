"""视觉检测 Agent —— 面向工业表面缺陷检测的自动化实验流水线。

设计目标：
    用户只描述「数据集类型 + 选用模型 + 工作目录」，
    Agent 负责调研数据集与论文、拉取官方模型代码、下载数据、
    转换标注、生成并执行训练（含消融）、搭建可视化系统、
    做推理测试、最后产出完整项目文档。

模块划分：
    config        任务配置（task.yaml）的数据模型与解析
    workspace     工作区目录规划与路径解析
    state         流水线状态机与产物清单（支持断点续跑）
    llm           大模型客户端（OpenAI 兼容 / Anthropic 协议 / 离线模板降级）
    envcheck      conda / anaconda / PyTorch / GPU 环境自检与解释器解析
    convert       标注格式转换（VOC XML / COCO JSON / YOLO txt）与数据集切分
    codegen       模型 YAML、训练脚本、可视化 App、推理脚本的代码生成
    search        权威数据集站点、学术检索、GitHub 官方仓库检索
    stages        S0-S8 九个阶段的实现
    pipeline      阶段编排、状态推进、失败恢复
    cli           命令行入口
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
