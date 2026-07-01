<div align="center">
  <a href="https://github.com/SakanaAI/AI-Scientist_v2/blob/main/docs/logo_v1.jpg">
    <img src="docs/logo_v1.png" width="215" alt="AI Scientist v2 Logo" />
  </a>
  <h1>
    <b>The AI Scientist-v2: 基于智能体树搜索的</b><br>
    <b>Workshop 级自动化科学发现</b>
  </h1>
</div>

<p align="center">
  📚 <a href="https://pub.sakana.ai/ai-scientist-v2/paper">[论文]</a> |
  📝 <a href="https://sakana.ai/ai-scientist-first-publication/">[博客文章]</a> |
  📂 <a href="https://github.com/SakanaAI/AI-Scientist-ICLR2025-Workshop-Experiment">[ICLR2025 Workshop 实验]</a>
</p>

全自动化的科学研究系统正变得越来越强大，AI 在推动科学发现方式的变革中扮演着关键角色。
我们激动地推出 The AI Scientist-v2——一个通用的端到端智能体系统，该系统已生成了首篇完全由 AI 撰写并通过同行评审的 workshop 论文。

该系统能够自主生成假设、运行实验、分析数据并撰写科学手稿。与[其前身（AI Scientist-v1）](https://github.com/SakanaAI/AI-Scientist)不同，AI Scientist-v2 不再依赖人工编写的模板，能够泛化到不同的机器学习（ML）领域，并采用由实验管理智能体引导的渐进式智能体树搜索方法。

> **注意：**
> AI Scientist-v2 并不一定比 v1 产出更好的论文，尤其是在已有强大的起始模板可用时。v1 遵循定义明确的模板，因此成功率高；而 v2 采用更广泛、更具探索性的方法，成功率较低。v1 最适合具有明确目标和坚实基础的场景，而 v2 则专为开放式科学探索而设计。

> **警告！**
> 此代码库将执行由大语言模型（LLM）编写的代码。这种自主性伴随着各种风险和挑战，包括可能使用危险软件包、不受控制的网络访问以及可能产生非预期的进程。请确保在受控的沙箱环境（例如 Docker 容器）中运行此代码。使用风险自负。

## 目录

1.  [环境要求](#环境要求)
    *   [安装](#安装)
    *   [支持的模型和 API 密钥](#支持的模型和-api-密钥)
2.  [生成研究想法](#生成研究想法)
3.  [运行 AI Scientist-v2 论文生成实验](#运行-ai-scientist-v2-论文生成实验)
4.  [引用 The AI Scientist-v2](#引用-the-ai-scientist-v2)
5.  [常见问题](#常见问题)
6.  [致谢](#致谢)

## 环境要求

本代码设计在配备 NVIDIA GPU 的 Linux 系统上运行，使用 CUDA 和 PyTorch。

### 安装

```bash
# 创建一个新的 conda 环境
conda create -n ai_scientist python=3.11
conda activate ai_scientist

# 安装支持 CUDA 的 PyTorch（请根据你的环境调整 pytorch-cuda 版本）
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# 安装 PDF 和 LaTeX 工具
conda install anaconda::poppler
conda install conda-forge::chktex

# 安装 Python 包依赖
pip install -r requirements.txt
```

安装过程通常不超过一小时。

### 支持的模型和 API 密钥

#### OpenAI 模型

默认情况下，系统使用 `OPENAI_API_KEY` 环境变量来访问 OpenAI 模型。

#### Gemini 模型

默认情况下，系统通过 OpenAI API 使用 `GEMINI_API_KEY` 环境变量来访问 Gemini 模型。

#### Claude 模型（通过 AWS Bedrock）

要使用由 Amazon Bedrock 提供的 Claude 模型，请安装必要的额外软件包：
```bash
pip install anthropic[bedrock]
```
接下来，配置有效的 [AWS 凭证](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-envvars.html)和目标 [AWS 区域](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html)，设置以下环境变量：`AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`、`AWS_REGION_NAME`。

#### Semantic Scholar API（文献搜索）

我们的代码可以选择使用 Semantic Scholar API 密钥（`S2_API_KEY`），以便在文献搜索期间获得更高的吞吐量（[如果你有的话](https://www.semanticscholar.org/product/api)）。该密钥在构思和论文撰写阶段都会使用。没有该密钥系统通常也能工作，但在构思阶段可能会遇到速率限制或新颖性检查能力下降。如果你在使用 Semantic Scholar 时遇到问题，可以在论文生成阶段跳过引用环节。

#### 设置 API 密钥

请确保为你计划使用的模型提供必要的 API 密钥作为环境变量。例如：
```bash
export OPENAI_API_KEY="YOUR_OPENAI_KEY_HERE"
export S2_API_KEY="YOUR_S2_KEY_HERE"
# 如果使用 Bedrock，请设置 AWS 凭证
# export AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
# export AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_KEY"
# export AWS_REGION_NAME="your-aws-region"
```

## 生成研究想法

在运行完整的 AI Scientist-v2 实验流水线之前，你需要先使用 `ai_scientist/perform_ideation_temp_free.py` 脚本来生成潜在的研究想法。该脚本使用 LLM 根据你提供的高层级主题描述来头脑风暴和精炼想法，并与 Semantic Scholar 等工具交互以检查新颖性。

1.  **准备主题描述：** 创建一个 Markdown 文件（例如 `my_research_topic.md`），描述你希望 AI 探索的研究领域或主题。该文件应包含 `Title`、`Keywords`、`TL;DR` 和 `Abstract` 等章节，以定义研究范围。请参考示例文件 `ai_scientist/ideas/i_cant_believe_its_not_better.md` 了解预期的结构和内容格式。将你的文件放在脚本可访问的位置（例如 `ai_scientist/ideas/` 目录）。

2.  **运行构思脚本：** 从主项目目录执行脚本，指定主题描述文件路径和所需的 LLM。

    ```bash
    python ai_scientist/perform_ideation_temp_free.py \
     --workshop-file "ai_scientist/ideas/my_research_topic.md" \
     --model gpt-4o-2024-05-13 \
     --max-num-generations 20 \
     --num-reflections 5
    ```
    *   `--workshop-file`：主题描述 Markdown 文件的路径。
    *   `--model`：用于生成想法的 LLM（请确保你已设置相应的 API 密钥）。
    *   `--max-num-generations`：尝试生成多少个不同的研究想法。
    *   `--num-reflections`：LLM 对每个想法进行多少次精炼步骤。

3.  **输出：** 脚本将生成一个 JSON 文件，文件名基于你的输入 Markdown 文件（例如 `ai_scientist/ideas/my_research_topic.json`）。该文件将包含一系列结构化的研究想法，包括假设、拟议的实验和相关工作分析。

4.  **进入实验阶段：** 获得包含研究想法的 JSON 文件后，你可以继续下一节来运行实验。

此构思步骤将 AI Scientist 引导至特定的兴趣领域，并产生具体的研究方向，供主实验流水线进行测试。

## 运行 AI Scientist-v2 论文生成实验

使用上一构思步骤中生成的 JSON 文件，你现在可以启动主 AI Scientist-v2 流水线。这包括通过智能体树搜索运行实验、分析结果并生成论文草稿。

通过命令行参数指定用于撰写和评审阶段的模型。
最佳优先树搜索（BFTS）的配置位于 `bfts_config.yaml` 中。请根据需要调整该文件中的参数。

`bfts_config.yaml` 中的关键树搜索配置参数：

-   `agent` 配置：
    -   设置 `num_workers`（并行探索路径的数量）和 `steps`（要探索的最大节点数）。例如，如果 `num_workers=3` 且 `steps=21`，树搜索将探索最多 21 个节点，每步并发扩展 3 个节点。
    -   `num_seeds`：如果 `num_workers` 小于 3，通常应与 `num_workers` 保持一致。否则，将 `num_seeds` 设置为 3。
    -   注意：其他 agent 参数如 `k_fold_validation`、`expose_prediction` 和 `data_preview` 在当前版本中未使用。
-   `search` 配置：
    -   `max_debug_depth`：agent 在放弃该搜索路径之前尝试调试失败节点的最大次数。
    -   `debug_prob`：尝试调试失败节点的概率。
    -   `num_drafts`：第一阶段中初始根节点的数量（即独立生长的树的数量）。

以下是使用生成的想法文件（如 `my_research_topic.json`）运行 AI-Scientist-v2 的示例命令。请查看 `bfts_config.yaml` 了解详细的树搜索参数（默认配置中实验部分使用 `claude-3-5-sonnet`）。如果你不想用代码片段初始化实验，请不要设置 `load_code`。

```bash
python launch_scientist_bfts.py \
 --load_ideas "ai_scientist/ideas/my_research_topic.json" \
 --load_code \
 --add_dataset_ref \
 --model_writeup o1-preview-2024-09-12 \
 --model_citation gpt-4o-2024-11-20 \
 --model_review gpt-4o-2024-11-20 \
 --model_agg_plots o3-mini-2025-01-31 \
 --num_cite_rounds 20
```

初始实验阶段完成后，你将在 `experiments/` 目录中找到一个带时间戳的日志文件夹。进入 `experiments/"timestamp_ideaname"/logs/0-run/` 文件夹，找到树可视化文件 `unified_tree_viz.html`。
所有实验阶段完成后，撰写阶段将开始。撰写阶段通常总共需要约 20 到 30 分钟。完成后，你应该能在 `timestamp_ideaname` 文件夹中看到 `timestamp_ideaname.pdf`。
对于此示例运行，所有阶段通常会在数小时内完成。

## 引用 The AI Scientist-v2

如果你在研究中使用了 **The AI Scientist-v2**，请按以下格式引用我们的工作：

```bibtex
@article{aiscientist_v2,
  title={The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search},
  author={Yamada, Yutaro and Lange, Robert Tjarko and Lu, Cong and Hu, Shengran and Lu, Chris and Foerster, Jakob and Clune, Jeff and Ha, David},
  journal={arXiv preprint arXiv:2504.08066},
  year={2025}
}
```

## 常见问题

**为什么我的实验没有生成 PDF 或评审结果？**

AI Scientist-v2 完成实验的成功率取决于所选用的基础模型以及想法的复杂程度。当使用强大的模型（如 Claude 3.5 Sonnet）进行实验阶段时，通常能获得更高的成功率。

**每次实验的预估成本是多少？**

构思步骤的成本取决于使用的 LLM 以及迭代次数和反思次数，但通常较低（几美元）。对于主实验流水线，使用 Claude 3.5 Sonnet 进行实验阶段通常每次运行花费约 $15–$20。随后的撰写阶段在使用示例命令中指定的默认模型时，额外增加约 $5。建议使用 GPT-4o 作为 `model_citation`，因为它有助于降低撰写成本。

**如何针对不同的学科领域运行 The AI Scientist-v2？**

首先，执行[生成研究想法](#生成研究想法)步骤。创建一个新的 Markdown 文件，描述你想要的学科领域或主题，遵循示例文件 `ai_scientist/ideas/i_cant_believe_its_not_better.md` 的结构。使用该文件运行 `perform_ideation_temp_free.py` 脚本以生成相应的 JSON 想法文件。然后，进入[运行 AI Scientist-v2 论文生成实验](#运行-ai-scientist-v2-论文生成实验)步骤，通过 `--load_ideas` 参数将此 JSON 文件与 `launch_scientist_bfts.py` 脚本一起使用。

**如果在访问 Semantic Scholar API 时遇到问题，该怎么办？**

Semantic Scholar API 用于评估所生成想法的新颖性，并在论文撰写阶段收集引用文献。如果你没有 API 密钥或遇到速率限制，你可能可以跳过这些阶段。

**我遇到了 "CUDA Out of Memory" 错误。该怎么办？**

此错误通常发生在 AI Scientist-v2 尝试加载或运行需要超出系统可用 GPU 显存的模型时。要解决此问题，你可以尝试更新构思提示文件（`ai_scientist/ideas/my_research_topic.md`），建议实验使用更小的模型。

## 致谢

`ai_scientist` 目录中实现的树搜索组件基于 [AIDE](https://github.com/WecoAI/aideml) 项目构建。我们感谢 AIDE 开发者的宝贵贡献以及他们将其工作公开发布。


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=SakanaAI/AI-Scientist-v2&type=Date)](https://star-history.com/#SakanaAI/AI-Scientist-v2&Date)

## ⚖️ 许可证与负责任使用

本项目基于 **The AI Scientist Source Code License**（衍生自 Responsible AI License）进行许可。

**强制披露：** 使用本代码，你具有法律义务在任何由此产生的科学手稿或论文中清晰、显著地披露 AI 的使用。

我们建议在论文的摘要或方法部分加入以下声明：
> "This manuscript was autonomously generated using [The AI Scientist](https://github.com/SakanaAI/AI-Scientist)."
