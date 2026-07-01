import argparse
import json
import os.path as osp
import re
import traceback
from typing import Any, Dict, List

import sys

sys.path.append(osp.join(osp.dirname(__file__), ".."))
from ai_scientist.llm import (
    AVAILABLE_LLMS,
    create_client,
    get_response_from_llm,
)

from ai_scientist.tools.semantic_scholar import SemanticScholarSearchTool
from ai_scientist.tools.base_tool import BaseTool

# Create tool instances
semantic_scholar_tool = SemanticScholarSearchTool()

# Define tools at the top of the file
tools = [
    semantic_scholar_tool,
    {
        "name": "FinalizeIdea",
        "description": """通过提供想法详情来最终确定你的想法。

IDEA JSON 应包含以下字段：
- "Name": 想法的简短描述。小写，无空格，可使用下划线。
- "Title": 一个吸引人且信息丰富的提案标题。
- "Short Hypothesis": 主假设或研究问题的简洁陈述。阐明此特定方向的必要性，确保这是研究此想法的最佳设置，并且没有其他明显更简单的方法来回答该问题。
- "Related Work": 对最相关的相关工作的简要讨论，以及提案如何与其明显区分，而非简单的扩展。
- "Abstract": 以会议格式总结提案的摘要（约 250 词）。
- "Experiments": 为验证提案而需要进行的一系列实验。确保这些实验简单且可行。具体说明你将如何测试假设，并详细描述精确的算法变更。包括你将使用的评估指标。
- "Risk Factors and Limitations": 提案的潜在风险和局限性列表。""",
    },
]

# Create a tools dictionary for easy lookup
tools_dict = {tool.name: tool for tool in tools if isinstance(tool, BaseTool)}

# Create a string with the tool descriptions
tool_descriptions = "\n\n".join(
    (
        f"- **{tool.name}**: {tool.description}"
        if isinstance(tool, BaseTool)
        else f"- **{tool['name']}**: {tool['description']}"
    )
    for tool in tools
)

# Extract tool names for the prompt
tool_names = [
    f'"{tool.name}"' if isinstance(tool, BaseTool) else f'"{tool["name"]}"'
    for tool in tools
]
tool_names_str = ", ".join(tool_names)

system_prompt = f"""你是一位经验丰富的AI研究员，旨在提出类似于激动人心的基金申请书的高影响力研究想法。请自由提出任何新颖的想法或实验；确保它们是新颖的。要有创造力，跳出思维框架。每个提案应源于关于该主题的一个简单而优雅的问题、观察或假设。例如，它们可以涉及非常有趣且简单的干预或调查，探索新的可能性或挑战现有假设。请清楚说明该提案如何与现有文献区分开来。

确保提案不需要超出学术实验室负担能力的资源。这些提案应能产生可在顶级机器学习会议上发表的论文。

你可以使用以下工具：

{tool_descriptions}

请按以下格式回复：

ACTION:
<要采取的行动，必须是 {tool_names_str} 之一>

ARGUMENTS:
<如果 ACTION 是 "SearchSemanticScholar"，请提供搜索查询 {{"query": "你的搜索查询"}}。如果 ACTION 是 "FinalizeIdea"，请提供想法详情 {{"idea": {{ ... }}}} 以及下面指定的 IDEA JSON。>

如果你选择最终确定你的想法，请在参数中提供 IDEA JSON：

IDEA JSON:
```json
{{
  "idea": {{
    "Name": "...",
    "Title": "...",
    "Short Hypothesis": "...",
    "Related Work": "...",
    "Abstract": "...",
    "Experiments": "...",
    "Risk Factors and Limitations": "..."
  }}
}}
```

确保 JSON 格式正确，以便自动解析。

注意：在最终确定你的想法之前，你应该至少进行一次文献搜索，以确保你的想法充分借鉴了现有研究。"""

# Define the initial idea generation prompt
idea_generation_prompt = """{workshop_description}

以下是你已经生成的提案：

'''
{prev_ideas_string}
'''

开始生成一个有趣且新颖的高层次研究提案，要与之前提出的提案不同。
"""

# Define the reflection prompt
idea_reflection_prompt = """第 {current_round}/{num_reflections} 轮。

在你的思考中，首先仔细考虑你刚刚创建的提案的质量、新颖性和可行性。
纳入你认为对评估提案重要的任何其他因素。
确保提案清晰简洁，且 JSON 格式正确。
不要让事情过于复杂。
在下一轮尝试中，尝试改进和完善你的提案。
除非有明显的严重问题，否则应坚持原始想法的核心精神。

如果你从工具中获得了新的信息，例如文献搜索结果，请将其纳入你的反思中，并相应地完善你的提案。

上一次操作的结果（如有）：

{last_tool_results}
"""


def generate_temp_free_idea(
    idea_fname: str,
    client: Any,
    model: str,
    workshop_description: str,
    max_num_generations: int = 20,
    num_reflections: int = 5,
    reload_ideas: bool = True,
) -> List[Dict]:
    idea_str_archive = []
    # load ideas from file
    if reload_ideas and osp.exists(idea_fname):
        with open(idea_fname, "r") as f:
            idea_str_content = json.load(f)
            for idea in idea_str_content:
                idea_str_archive.append(json.dumps(idea))
            print(f"已加载 {len(idea_str_archive)} 个创意，来自 {idea_fname}")
    else:
        print(f"在 {idea_fname} 中未找到创意。从头开始。")

    for gen_idx in range(max_num_generations):
        print()
        print(f"正在生成提案 {gen_idx + 1}/{max_num_generations}")
        try:
            prev_ideas_string = "\n\n".join(idea_str_archive)

            last_tool_results = ""
            idea_finalized = False
            msg_history = []

            for reflection_round in range(num_reflections):
                if reflection_round == 0:
                    # Use the initial idea generation prompt
                    prompt_text = idea_generation_prompt.format(
                        workshop_description=workshop_description,
                        prev_ideas_string=prev_ideas_string,
                    )
                else:
                    # Use the reflection prompt, including tool results if any
                    prompt_text = idea_reflection_prompt.format(
                        current_round=reflection_round + 1,
                        num_reflections=num_reflections,
                        last_tool_results=last_tool_results or "No new results.",
                    )

                response_text, msg_history = get_response_from_llm(
                    prompt=prompt_text,
                    client=client,
                    model=model,
                    system_message=system_prompt,
                    msg_history=msg_history,
                )

                # Parse the LLM's response
                try:
                    # Use regular expressions to extract the components
                    action_pattern = r"ACTION:\s*(.*?)\s*ARGUMENTS:"
                    arguments_pattern = r"ARGUMENTS:\s*(.*?)(?:$|\nTHOUGHT:|\n$)"

                    action_match = re.search(
                        action_pattern, response_text, re.DOTALL | re.IGNORECASE
                    )
                    arguments_match = re.search(
                        arguments_pattern, response_text, re.DOTALL | re.IGNORECASE
                    )

                    if not all([action_match, arguments_match]):
                        raise ValueError("Failed to parse the LLM response.")

                    action = action_match.group(1).strip()
                    arguments_text = arguments_match.group(1).strip()
                    print(f"动作: {action}")
                    print(f"参数: {arguments_text}")

                    # If arguments are wrapped in ```json blocks, extract the content
                    if arguments_text.startswith("```json"):
                        arguments_text = re.search(
                            r"```json\s*(.*?)\s*```", arguments_text, re.DOTALL
                        ).group(1)

                    # Process the action and arguments
                    if action in tools_dict:
                        # It's a tool we have defined
                        tool = tools_dict[action]
                        # Parse arguments
                        try:
                            arguments_json = json.loads(arguments_text)
                        except json.JSONDecodeError:
                            raise ValueError(f"Invalid arguments JSON for {action}.")

                        # Use the tool
                        try:
                            # Assuming the arguments match the parameters of the tool
                            result = tool.use_tool(**arguments_json)
                            last_tool_results = result
                        except Exception as e:
                            last_tool_results = f"Error using tool {action}: {str(e)}"
                    elif action == "FinalizeIdea":
                        # Parse arguments
                        try:
                            arguments_json = json.loads(arguments_text)
                            idea = arguments_json.get("idea")
                            if not idea:
                                raise ValueError("Missing 'idea' in arguments.")

                            # Append the idea to the archive
                            idea_str_archive.append(json.dumps(idea))
                            print(f"提案已完成: {idea}")
                            idea_finalized = True
                            break
                        except json.JSONDecodeError:
                            raise ValueError("Invalid arguments JSON for FinalizeIdea.")
                    else:
                        print(
                            "无效的动作。请指定一个可用工具。"
                        )
                        print(f"可用的动作有: {tool_names_str}")
                except Exception as e:
                    print(
                        f"无法解析 LLM 回复。回复文本:\n{response_text}"
                    )
                    traceback.print_exc()
                    break  # Exit the loop if parsing fails

            if idea_finalized:
                continue  # Move to the next idea

        except Exception as e:
            print("生成提案失败:")
            traceback.print_exc()
            continue

    # Save ideas
    ideas = [json.loads(idea_str) for idea_str in idea_str_archive]

    with open(idea_fname, "w") as f:
        json.dump(ideas, f, indent=4)
    print(f"已将 {len(ideas)} 个创意保存至 {idea_fname}")
    return ideas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate AI scientist proposals - template free"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--max-num-generations",
        type=int,
        default=1,
        help="Maximum number of proposal generations.",
    )
    parser.add_argument(
        "--workshop-file",
        type=str,
        default="ideas/i_cant_believe_its_not_better.md",
        help="Path to the workshop description file.",
    )
    parser.add_argument(
        "--num-reflections",
        type=int,
        default=5,
        help="Number of reflection rounds per proposal.",
    )
    args = parser.parse_args()

    # Create the LLM client
    client, client_model = create_client(args.model)

    with open(args.workshop_file, "r") as f:
        workshop_description = f.read()
    print(f"使用来自 {args.workshop_file} 的研讨会描述进行创意生成。")
    print(f"研讨会描述:\n{workshop_description}")

    # Create output filename by replacing .md extension with .json
    idea_fname = args.workshop_file.replace(".md", ".json")
    print("开始为", idea_fname, "生成创意")
    ideas = generate_temp_free_idea(
        idea_fname=idea_fname,
        client=client,
        model=client_model,
        workshop_description=workshop_description,
        max_num_generations=args.max_num_generations,
        num_reflections=args.num_reflections,
    )
    print(f"{args.workshop_file} 已生成 {len(ideas)} 个创意。")
