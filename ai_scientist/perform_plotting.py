import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from rich import print

from ai_scientist.llm import create_client, get_response_from_llm
from ai_scientist.utils.token_tracker import token_tracker
from ai_scientist.perform_icbinb_writeup import (
    load_idea_text,
    load_exp_summaries,
    filter_experiment_summaries,
)

MAX_FIGURES = 12

AGGREGATOR_SYSTEM_MSG = f"""你是一位雄心勃勃的AI研究员，正在为科学论文投稿准备最终图表。
你有多个实验摘要（baseline、research、ablation），每个可能包含对不同图表或数值洞察的引用。
还有一个顶层 'research_idea.md' 文件，概述了总体研究方向。
你的任务是生成一个 Python 脚本，全面聚合和可视化最终结果，用于一篇完整的研究论文。

关键要点：
1) 合并或复现相关的现有绘图代码，参考数据最初是如何生成的（通过代码引用）以确保正确性。
2) 创建完整的最终科学图表集，仅存储在 'figures/' 中（因为只有这些会被用于最终论文）。
3) 确保使用现有的 .npy 数据进行分析；不要虚构数据。如果需要单一数值结果，可以从 JSON 摘要中复制。
4) 只创建最适合以图形而非表格呈现数据的图表。例如，如果数据难以视觉比较，不要使用柱状图。
5) 最终的聚合脚本必须放在三个反引号中，并且能够独立运行，可以放入代码库直接执行。
6) 如果存在基于合成数据的图表，将其包含在附录中。

实现最佳实践：
- 不要生成多余或不相关的图表。
- 保持清晰，代码简洁但充分。
- 展示最终研究论文投稿所需的全面性。
- 不要引用不存在的文件或图像。
- 使用 .npy 文件获取图表数据，使用 JSON 摘要获取关键数值。
- 为每个图表划定边界，并将它们放在单独的 try-catch 块中，确保一个图表的失败不会影响其他图表。
- 确保只创建最终论文和附录所需且独特的图表。总共约 {MAX_FIGURES} 个图表是比较合适的数量。
- 如果合适，将多个图表聚合到一个图形中，即如果它们都与同一主题相关。你可以在一行中放置最多 3 个子图。
- 提供标注清晰的图表（坐标轴、图例、标题），突出主要发现。在所有地方使用信息丰富的名称，包括图例中的名称，以便在最终论文中引用。确保图例始终可见。
- 使图表看起来专业（如适用，去除顶部和右侧边框，dpi 设为 300，合适的 ylim 等）。
- 不要在标签中使用下划线，例如 "loss_vs_epoch" 应为 "loss vs epoch"。
- 对于图像示例，选择多个类别来展示结果的多样性，而不是只展示单一类别。部分可放入正文，其余放入附录。

你的输出应为完整的 Python 聚合脚本，放在三个反引号中。
"""


def build_aggregator_prompt(combined_summaries_str, idea_text):
    return f"""
我们有三个科学实验的 JSON 摘要：baseline、research、ablation。
它们可能包含图表描述列表、生成图表的代码以及指向包含数值结果的 .npy 文件的路径。
我们的目标是生成最终的、可发表的图表。

--- 研究思路 ---
```
{idea_text}
```

重要提示：
- 聚合脚本必须从 "exp_results_npy_files" 字段中加载现有的 .npy 实验数据（仅使用摘要 JSON 中完整且精确的文件路径）以进行全面的绘图。
- 在保存任何图表之前，应调用 os.makedirs("figures", exist_ok=True)。
- 力求在 'figures/' 中平衡实证结果、消融实验和多样化的信息丰富的可视化，全面展示最终的研究成果。
- 如果你需要摘要中的 .npy 路径，只需直接复制这些路径（而不是复制和解析整个摘要）。

你生成的 Python 脚本必须：
1) 从这些摘要中加载或引用相关数据和 .npy 文件。使用摘要 JSON 中完整且精确的文件路径。
2) 综合或直接创建最终论文所需的、具有科学意义的最终图表（全面且完整），如果需要，参考原始代码以了解数据是如何生成的。
3) 仔细组合或复现相关现有绘图代码，仅在 'figures/' 中生成这些最终聚合图表，因为只有这些会被用于最终论文。
4) 不要虚构数据。数据必须从 .npy 文件加载或从 JSON 摘要中复制。
5) 聚合脚本必须是完全自包含的，并将最终图表放置在 'figures/' 中。
6) 此聚合脚本应为最终论文生成一套全面且最终的科研图表，反映实验数据中的所有主要发现。
7) 确保每个图表都是独特的，不与原始图表重复。如有必要，删除任何重复的图表。
8) 每个图形最多可包含 3 个子图，使用 fig, ax = plt.subplots(1, 3)。
9) 对图表标签和标题使用比默认更大的字体大小，以确保在最终 PDF 论文中可读。


以下是 JSON 格式的摘要：

{combined_summaries_str}

请在三个反引号中返回一个 Python 脚本。
"""


def extract_code_snippet(text: str) -> str:
    """
    Look for a Python code block in triple backticks in the LLM response.
    Return only that code. If no code block is found, return the entire text.
    """
    pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return matches[0].strip() if matches else text.strip()


def run_aggregator_script(
    aggregator_code, aggregator_script_path, base_folder, script_name
):
    if not aggregator_code.strip():
        print("No aggregator code was provided. Skipping aggregator script run.")
        return ""
    with open(aggregator_script_path, "w") as f:
        f.write(aggregator_code)

    print(
        f"Aggregator script written to '{aggregator_script_path}'. Attempting to run it..."
    )

    aggregator_out = ""
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=base_folder,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        aggregator_out = result.stdout + "\n" + result.stderr
        print("Aggregator script ran successfully.")
    except subprocess.CalledProcessError as e:
        aggregator_out = (e.stdout or "") + "\n" + (e.stderr or "")
        print("Error: aggregator script returned a non-zero exit code.")
        print(e)
    except Exception as e:
        aggregator_out = str(e)
        print("Error while running aggregator script.")
        print(e)

    return aggregator_out


def aggregate_plots(
    base_folder: str, model: str = "o1-2024-12-17", n_reflections: int = 5
) -> None:
    filename = "auto_plot_aggregator.py"
    aggregator_script_path = os.path.join(base_folder, filename)
    figures_dir = os.path.join(base_folder, "figures")

    # Clean up previous files
    if os.path.exists(aggregator_script_path):
        os.remove(aggregator_script_path)
    if os.path.exists(figures_dir):
        shutil.rmtree(figures_dir)
        print(f"Cleaned up previous figures directory")

    idea_text = load_idea_text(base_folder)
    exp_summaries = load_exp_summaries(base_folder)
    filtered_summaries_for_plot_agg = filter_experiment_summaries(
        exp_summaries, step_name="plot_aggregation"
    )
    # Convert them to one big JSON string for context
    combined_summaries_str = json.dumps(filtered_summaries_for_plot_agg, indent=2)

    # Build aggregator prompt
    aggregator_prompt = build_aggregator_prompt(combined_summaries_str, idea_text)

    # Call LLM
    client, model_name = create_client(model)
    response, msg_history = None, []
    try:
        response, msg_history = get_response_from_llm(
            prompt=aggregator_prompt,
            client=client,
            model=model_name,
            system_message=AGGREGATOR_SYSTEM_MSG,
            print_debug=False,
            msg_history=msg_history,
        )
    except Exception:
        traceback.print_exc()
        print("Failed to get aggregator script from LLM.")
        return

    aggregator_code = extract_code_snippet(response)
    if not aggregator_code.strip():
        print(
            "No Python code block was found in LLM response. Full response:\n", response
        )
        return

    # First run of aggregator script
    aggregator_out = run_aggregator_script(
        aggregator_code, aggregator_script_path, base_folder, filename
    )

    # Multiple reflection loops
    for i in range(n_reflections):
        # Check number of figures
        figure_count = 0
        if os.path.exists(figures_dir):
            figure_count = len(
                [
                    f
                    for f in os.listdir(figures_dir)
                    if os.path.isfile(os.path.join(figures_dir, f))
                ]
            )
        print(f"[{i + 1} / {n_reflections}]: Number of figures: {figure_count}")
        # Reflection prompt with reminder for common checks and early exit
        reflection_prompt = f"""我们已运行你的聚合脚本，它生成了 {figure_count} 个图表。脚本的输出如下：
```
{aggregator_out}
```

请对当前脚本提出批评，指出任何缺陷，包括但不限于：
- 这些图表是否足够用于最终论文投稿？不要创建超过 {MAX_FIGURES} 个图表。
- 你确保既使用了关键数值，又从 .npy 文件生成了更详细的图表吗？
- 图表标题和图例是否使用了信息丰富且描述性的名称？这些图表是最终版本，确保没有注释或其他备注。
- 如果合适，可以将多个图表聚合到一个图形中吗？
- 标签中是否包含下划线？如果有，请替换为空格。
- 确保每个图表都是独特的，不与原始图表重复。

如果你认为已完成，只需说："I am done"。否则，请在三个反引号中提供更新后的聚合脚本。"""

        print("[green]Reflection prompt:[/green] ", reflection_prompt)
        try:
            reflection_response, msg_history = get_response_from_llm(
                prompt=reflection_prompt,
                client=client,
                model=model_name,
                system_message=AGGREGATOR_SYSTEM_MSG,
                print_debug=False,
                msg_history=msg_history,
            )

        except Exception:
            traceback.print_exc()
            print("Failed to get reflection from LLM.")
            return

        # Early-exit check
        if figure_count > 0 and "I am done" in reflection_response:
            print("LLM indicated it is done with reflections. Exiting reflection loop.")
            break

        aggregator_new_code = extract_code_snippet(reflection_response)

        # If new code is provided and differs, run again
        if (
            aggregator_new_code.strip()
            and aggregator_new_code.strip() != aggregator_code.strip()
        ):
            aggregator_code = aggregator_new_code
            aggregator_out = run_aggregator_script(
                aggregator_code, aggregator_script_path, base_folder, filename
            )
        else:
            print(
                f"No new aggregator script was provided or it was identical. Reflection step {i+1} complete."
            )


def main():
    parser = argparse.ArgumentParser(
        description="Generate and execute a final plot aggregation script with LLM assistance."
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Path to the experiment folder with summary JSON files.",
    )
    parser.add_argument(
        "--model",
        default="o1-2024-12-17",
        help="LLM model to use (default: o1-2024-12-17).",
    )
    parser.add_argument(
        "--reflections",
        type=int,
        default=5,
        help="Number of reflection steps to attempt (default: 5).",
    )
    args = parser.parse_args()
    aggregate_plots(
        base_folder=args.folder, model=args.model, n_reflections=args.reflections
    )


if __name__ == "__main__":
    main()
