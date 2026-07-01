# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language / 语言

本项目已完成中文化改造。所有 LLM 提示词（系统消息、角色设定、指令）均已翻译为中文，AI 在实验过程中的思考、分析、总结均使用中文输出。最终生成的学术论文仍为英文（符合国际学术出版标准）。README.md 也已翻译为中文。

## Project Overview

The AI Scientist-v2 by Sakana AI is a fully autonomous scientific research system that generates hypotheses, runs ML experiments, analyzes data, and writes complete scientific manuscripts — all driven by LLMs. It produced the first AI-written workshop paper accepted through peer review (ICLR 2025). The core innovation over v1 is replacing human-authored templates with a progressive agentic Best-First Tree Search (BFTS) guided by an experiment manager agent. The tree search component is built on top of the [AIDE](https://github.com/WecoAI/aideml) project.

## Environment & Dependencies

- **OS**: Linux with NVIDIA GPUs (CUDA, PyTorch)
- **Python**: 3.11
- **Package manager**: conda + pip from `requirements.txt`
- **No build/lint/test tooling** exists in this project (research codebase)

## Key Commands

### Idea Generation (Stage 0)
```bash
python ai_scientist/perform_ideation_temp_free.py \
  --workshop-file "ai_scientist/ideas/my_research_topic.md" \
  --model gpt-4o-2024-05-13 \
  --max-num-generations 20 \
  --num-reflections 5
```
Generates a JSON file of structured research ideas from a topic markdown. The input markdown should have sections: `Title`, `Keywords`, `TL;DR`, `Abstract`. See `ai_scientist/ideas/i_cant_believe_its_not_better.md` for the expected format.

### Full Experiment Pipeline
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
Runs the full pipeline: BFTS experiments → plot aggregation → paper writeup → AI review. Results go to `experiments/<timestamp>_<idea_name>/`. Use `--skip_writeup` / `--skip_review` to run only experiments. Use `--writeup-type normal` for 8-page ICML format (default is `icbinb` for 4-page).

### Code formatting
```bash
black .
```
`black` is the only dev dependency listed in `requirements.txt`.

## Architecture

### Pipeline Flow

```
Stage 0: Ideation (perform_ideation_temp_free.py)
    → Generates research ideas from topic description, using Semantic Scholar API for literature search

Stage 1-4: BFTS Experiments (launch_scientist_bfts.py → perform_experiments_bfts)
    → AgentManager runs 4 sequential stages across solution trees:
      1. initial_implementation — draft working code from scratch
      2. baseline_tuning — hyperparameter optimization
      3. creative_research — novel improvements beyond baseline
      4. ablation_studies — component contribution analysis
    → Each stage iterates via ParallelAgent using Best-First Tree Search

Stage 5: Plot Aggregation (perform_plotting.py)
    → Merges all experiment-stage plots into publishable figures

Stage 6: Paper Writeup (perform_writeup.py / perform_icbinb_writeup.py)
    → Generates LaTeX paper from experiment summaries + citations

Stage 7: AI Review (perform_llm_review.py + perform_vlm_review.py)
    → LLM reviews paper text; VLM reviews figures/captions
```

### Core Component Hierarchy

**`AgentManager`** (`treesearch/agent_manager.py`) — Top-level experiment manager. Creates `ParallelAgent` instances per stage, uses LLMs to generate sub-stage goals, evaluate completion, and decide when to progress. Seeds each stage with the best implementation from the previous stage.

**`ParallelAgent`** (`treesearch/parallel_agent.py`) — Per-stage workhorse. Manages a `ProcessPoolExecutor` with GPU-aware worker assignment. Implements Best-First Tree Search: selects nodes to expand (draft, debug, improve, hyperparam-tune, ablate) and evaluates results.

**`MinimalAgent`** — Runs inside each worker process. Generates code via LLM (`_draft`, `_debug`, `_improve`, `_generate_hyperparam_tuning_node`, `_generate_ablation_node`), executes it via `Interpreter`, parses results via LLM, generates plotting code, and analyzes plots via VLM.

**`Journal` + `Node`** (`treesearch/journal.py`) — Solution tree data structure. Nodes form a DAG (parent-child). Journal provides query methods (`good_nodes`, `buggy_nodes`, `get_best_node`). Best node selection can use LLM-based analysis.

**`Interpreter`** (`treesearch/interpreter.py`) — Sandboxed Python executor. Runs LLM-generated code in a child process with configurable timeout, capturing stdout/stderr/exceptions.

### LLM Integration

- **`ai_scientist/llm.py`** — Unified client factory (`create_client(model)`) supporting OpenAI (GPT-4o, o1, o3), Anthropic (Claude via direct API / Bedrock / Vertex AI), Google Gemini, DeepSeek, Ollama (various open models like Qwen, Llama). Also provides `get_response_from_llm()` and `get_batch_responses_from_llm()`.
- **`ai_scientist/vlm.py`** — Vision-language model client for analyzing experiment plots.
- **`treesearch/backend/`** — LLM query dispatcher. Routes to OpenAI backend (with function/tool calling support via `FunctionSpec`) or Anthropic backend (text only, no function calling), based on model name.

### Configuration

- **`bfts_config.yaml`** — Central config for tree search. Key sections:
  - `agent.code.model` — LLM for code generation (default: Claude 3.5 Sonnet via Bedrock)
  - `agent.feedback.model` — LLM for evaluating program output
  - `agent.vlm_feedback.model` — VLM for plot analysis
  - `agent.num_workers` — parallel exploration paths
  - `agent.steps` — max nodes to explore per stage
  - `agent.search.max_debug_depth` — max debug attempts before abandoning a path
  - `agent.search.debug_prob` — probability of debugging a failing node
  - `agent.search.num_drafts` — number of initial root nodes in Stage 1
  - `agent.multi_seed_eval.num_seeds` — seeds for multi-seed evaluation

- **`report.model`** — LLM for final journal-to-report conversion. Override via `--model_writeup` and `--model_review` CLI args.

### Experiment Output

Each run creates `experiments/<timestamp>_<idea_name>_attempt_<N>/`:
- `logs/0-run/unified_tree_viz.html` — tree search visualization (available after Stage 1)
- `<timestamp>_<idea_name>.pdf` — final generated paper
- `review_text.txt`, `review_img_cap_ref.json` — AI review results
- `token_tracker.json` — token usage and cost summary

## API Keys

Required environment variables (depending on models used):
- `OPENAI_API_KEY` — OpenAI models
- `GEMINI_API_KEY` — Gemini models (via OpenAI API)
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION_NAME` — Bedrock Claude models
- `S2_API_KEY` — Semantic Scholar (optional, for higher throughput)
- `DEEPSEEK_API_KEY` — DeepSeek models
- `HUGGINGFACE_API_KEY` — DeepCoder model
- `OPENROUTER_API_KEY` — Llama 3.1 via OpenRouter

## Important Notes

- This codebase **executes LLM-generated Python code**. Always run within a sandboxed environment (Docker container recommended).
- Estimated cost: ~$15-20 for experiments (Claude 3.5 Sonnet) + ~$5 for writeup per run.
- The LICENSE requires **mandatory disclosure** of AI use in any resulting scientific manuscripts.
- Claude models via Anthropic backend do NOT support function/tool calling (only OpenAI backend does). The `FunctionSpec` mechanism in `treesearch/backend/utils.py` is only used with OpenAI models.
- The v1 codebase is at `SakanaAI/AI-Scientist` (separate repo). v2 removes template dependence but has lower success rates — it's designed for open-ended exploration, not tasks with well-defined templates.
