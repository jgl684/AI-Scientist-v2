from .backend import query
from .journal import Journal
from .utils.config import StageConfig


def journal2report(journal: Journal, task_desc: dict, rcfg: StageConfig):
    """
    Generate a report from a journal, the report will be in markdown format.
    """
    report_input = journal.generate_summary(include_code=True)
    system_prompt_dict = {
        "Role": "你是一位总是使用简洁语言的研究助理。",
        "Goal": "目标是撰写一份技术报告，总结实证发现和技术决策。",
        "Input": "你会收到一份原始的研究日志，其中包含设计尝试及其结果的列表，以及研究思路描述。",
        "Output": [
            "你的输出应为一份单一的markdown文档。",
            "你的报告应包含以下章节：Introduction, Preprocessing, Methods, Results Discussion, Future Work",
            "你可以根据需要包含子章节。",
        ],
    }
    context_prompt = (
        f"Here is the research journal of the agent: <journal>{report_input}<\\journal>, "
        f"and the research idea description is: <research_proposal>{task_desc}<\\research_proposal>."
    )
    return query(
        system_message=system_prompt_dict,
        user_message=context_prompt,
        model=rcfg.model,
        temperature=rcfg.temp,
        max_tokens=4096,
    )
