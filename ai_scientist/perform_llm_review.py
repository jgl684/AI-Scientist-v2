import os
import json
import numpy as np
from pypdf import PdfReader
import pymupdf
import pymupdf4llm
from ai_scientist.llm import (
    get_response_from_llm,
    get_batch_responses_from_llm,
    extract_json_between_markers,
)

reviewer_system_prompt_base = (
    "你是一位正在审阅提交给顶级机器学习会议的论文的AI研究员。"
    "请对你的决定持批判和谨慎态度。"
)
reviewer_system_prompt_neg = (
    reviewer_system_prompt_base
    + "如果论文质量差或者你不确定，给它低分并拒绝它。"
)
reviewer_system_prompt_pos = (
    reviewer_system_prompt_base
    + "如果论文质量好或者你不确定，给它高分并接受它。"
)

template_instructions = """
请按照以下格式回复：

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

在 <THOUGHT> 中，首先简要讨论你对评估的直觉和推理。
详细阐述你的高层次论点、必要选择以及审阅的期望结果。
不要在此处发表泛泛的评论，要针对你当前审阅的论文具体说明。
将此视为审阅的笔记阶段。

在 <JSON> 中，按顺序以 JSON 格式提供审阅，包含以下字段：
- "Summary": 论文内容及其贡献的摘要。
- "Strengths": 论文的优点列表。
- "Weaknesses": 论文的缺点列表。
- "Originality": 1 到 4 的评分（低、中、高、非常高）。
- "Quality": 1 到 4 的评分（低、中、高、非常高）。
- "Clarity": 1 到 4 的评分（低、中、高、非常高）。
- "Significance": 1 到 4 的评分（低、中、高、非常高）。
- "Questions": 一组需要论文作者回答的澄清性问题。
- "Limitations": 一组工作的局限性和潜在的负面社会影响。
- "Ethical Concerns": 一个布尔值，指示是否存在伦理问题。
- "Soundness": 1 到 4 的评分（差、一般、好、优秀）。
- "Presentation": 1 到 4 的评分（差、一般、好、优秀）。
- "Contribution": 1 到 4 的评分（差、一般、好、优秀）。
- "Overall": 1 到 10 的评分（非常强烈拒绝 到 获奖质量）。
- "Confidence": 1 到 5 的评分（低、中、高、非常高、绝对确定）。
- "Decision": 必须是以下之一：Accept（接受）、Reject（拒绝）。

对于 "Decision" 字段，不要使用 Weak Accept、Borderline Accept、Borderline Reject 或 Strong Reject。只使用 Accept 或 Reject。
此 JSON 将自动解析，请确保格式精确。
"""

neurips_form = (
    """
## 审阅表格
以下描述了你在每篇论文审阅表格上会被问到的问题，以及回答这些问题时需要考虑的一些指导原则。
在撰写审阅意见时，请注意在做出决定后，被接受的论文和选择公开的被拒论文的审阅和元审阅将被公开发布。

1. 摘要（Summary）：简要总结论文及其贡献。这不是批评论文的地方；作者应该会同意一篇写得好的摘要。
  - 优点与缺点（Strengths and Weaknesses）：请对论文的优点和缺点进行全面评估，涵盖以下每个维度：
  - 原创性（Originality）：任务或方法是否新颖？这项工作是否是已知技术的新颖组合？（这可能很有价值！）是否清楚地说明了这项工作与先前贡献的区别？相关工作是否充分引用？
  - 质量（Quality）：提交的论文在技术上是否可靠？论点是否有充分支持（例如通过理论分析或实验结果）？使用的方法是否适当？这是一项完整的工作还是进展中的工作？作者是否谨慎且诚实地评估了其工作的优缺点？
  - 清晰性（Clarity）：提交的论文是否表达清晰？是否结构良好？（如果不是，请提供建设性的改进建议。）是否充分地向读者传达了信息？（请注意，一篇写得极好的论文应能为专业读者提供足够的信息来复现其结果。）
  - 重要性（Significance）：结果是否重要？其他人（研究人员或从业者）是否可能会使用这些想法或在此基础上进行构建？该论文是否以比先前工作更好的方式解决了一项困难任务？它是否以可证明的方式推动了技术前沿？它是否提供了独特的数据、关于现有数据的独特结论，或独特的理论或实验方法？

2. 问题（Questions）：请列出并仔细描述对作者的任何问题和建议。思考那些作者的回复可能会改变你的观点、澄清困惑或解决局限性的问题。这对于与作者进行富有成效的反驳和讨论阶段非常重要。

3. 局限性（Limitations）：作者是否充分说明了其工作的局限性和潜在的负面社会影响？如果没有，请提供建设性的改进建议。
一般来说，作者应当因坦诚面对其工作的局限性和任何潜在的负面社会影响而得到奖励，而非惩罚。鼓励你思考是否有任何关键点被遗漏，并将其作为反馈提供给作者。

4. 伦理问题（Ethical Concerns）：如果这篇论文存在伦理问题，请标记该论文以进行伦理审查。关于何时需要这样做，请参考 NeurIPS 伦理指南。

5. 可靠性（Soundness）：请按以下等级为论文分配一个数值评分，以表示技术主张、实验和研究方法的可靠性，以及论文的核心主张是否得到充分的证据支持。
  4: 优秀
  3: 良好
  2: 一般
  1: 差

6. 表述（Presentation）：请按以下等级为论文分配一个数值评分，以表示表述的质量。这应考虑写作风格和清晰度，以及相对于先前工作的背景阐述。
  4: 优秀
  3: 良好
  2: 一般
  1: 差

7. 贡献（Contribution）：请按以下等级为论文分配一个数值评分，以表示该论文对所研究领域的整体贡献质量。所提出的问题是否重要？论文是否在思想和/或执行方面具有显著的原创性？结果是否值得与更广泛的 NeurIPS 社区分享？
  4: 优秀
  3: 良好
  2: 一般
  1: 差

8. 总体评价（Overall）：请为本次提交提供一个"总体评分"。选项如下：
  10: 获奖质量：技术上完美的论文，对 AI 的一个或多个领域产生突破性影响，具有非常强的评估、可复现性和资源，且没有未解决的伦理问题。
  9: 非常强烈接受：技术上完美的论文，至少在 AI 的一个领域产生突破性影响，并在 AI 的多个领域产生卓越影响，具有完美的评估、资源和可复现性，且没有未解决的伦理问题。
  8: 强烈接受：技术上很强的论文，具有新颖的想法，至少在 AI 的一个领域产生卓越影响或在 AI 的多个领域产生高到卓越的影响，具有优秀的评估、资源和可复现性，且没有未解决的伦理问题。
  7: 接受：技术上可靠的论文，至少在 AI 的一个子领域产生高影响或在多个领域产生中到高影响，具有良好到优秀的评估、资源、可复现性，且没有未解决的伦理问题。
  6: 弱接受：技术上可靠、中到高影响的论文，在评估、资源、可复现性、伦理问题方面没有重大问题。
  5: 边界接受：技术上可靠的论文，接受的理由超过拒绝的理由，例如评估有限。请谨慎使用。
  4: 边界拒绝：技术上可靠的论文，拒绝的理由（例如评估有限）超过接受的理由（例如评估良好）。请谨慎使用。
  3: 拒绝：例如，存在技术缺陷、评估薄弱、可复现性不足和伦理问题未完全解决的论文。
  2: 强烈拒绝：例如，存在重大技术缺陷和/或评估差、影响有限、可复现性差以及伦理问题大部分未解决的论文。
  1: 非常强烈拒绝：例如，结果微不足道或伦理问题未解决的论文。

9. 置信度（Confidence）：请为你的评估提供一个"置信度评分"，以表示你对评估的确定程度。选项如下：
  5: 你对自己的评估完全确定。你非常熟悉相关工作，并仔细检查了数学/其他细节。
  4: 你对评估有信心，但并非绝对确定。不太可能（但并非不可能）你没有理解提交论文的某些部分，或者你不熟悉某些相关工作。
  3: 你对评估相当有信心。有可能你没有理解提交论文的某些部分，或者你不熟悉某些相关工作。数学/其他细节未被仔细检查。
  2: 你愿意为你的评估辩护，但很可能你没有理解提交论文的核心部分，或者你不熟悉某些相关工作。数学/其他细节未被仔细检查。
  1: 你的评估是一种有根据的猜测。该提交不在你的领域内，或者提交内容难以理解。数学/其他细节未被仔细检查。
"""
    + template_instructions
)


def perform_review(
    text,
    model,
    client,
    num_reflections=1,
    num_fs_examples=1,
    num_reviews_ensemble=1,
    temperature=0.75,
    msg_history=None,
    return_msg_history=False,
    reviewer_system_prompt=reviewer_system_prompt_neg,
    review_instruction_form=neurips_form,
):
    if num_fs_examples > 0:
        fs_prompt = get_review_fewshot_examples(num_fs_examples)
        base_prompt = review_instruction_form + fs_prompt
    else:
        base_prompt = review_instruction_form

    base_prompt += f"""
Here is the paper you are asked to review:
```
{text}
```"""

    if num_reviews_ensemble > 1:
        llm_reviews, msg_histories = get_batch_responses_from_llm(
            base_prompt,
            model=model,
            client=client,
            system_message=reviewer_system_prompt,
            print_debug=False,
            msg_history=msg_history,
            temperature=0.75,
            n_responses=num_reviews_ensemble,
        )
        parsed_reviews = []
        for idx, rev in enumerate(llm_reviews):
            try:
                parsed_reviews.append(extract_json_between_markers(rev))
            except Exception as e:
                print(f"Ensemble review {idx} failed: {e}")
        parsed_reviews = [r for r in parsed_reviews if r is not None]
        review = get_meta_review(model, client, temperature, parsed_reviews)
        if review is None:
            review = parsed_reviews[0]
        for score, limits in [
            ("Originality", (1, 4)),
            ("Quality", (1, 4)),
            ("Clarity", (1, 4)),
            ("Significance", (1, 4)),
            ("Soundness", (1, 4)),
            ("Presentation", (1, 4)),
            ("Contribution", (1, 4)),
            ("Overall", (1, 10)),
            ("Confidence", (1, 5)),
        ]:
            scores = []
            for r in parsed_reviews:
                if score in r and limits[0] <= r[score] <= limits[1]:
                    scores.append(r[score])
            if scores:
                review[score] = int(round(np.mean(scores)))
        msg_history = msg_histories[0][:-1]
        msg_history += [
            {
                "role": "assistant",
                "content": f"""
THOUGHT:
I will start by aggregating the opinions of {num_reviews_ensemble} reviewers that I previously obtained.

REVIEW JSON:
```json
{json.dumps(review)}
```
""",
            }
        ]
    else:
        llm_review, msg_history = get_response_from_llm(
            base_prompt,
            model=model,
            client=client,
            system_message=reviewer_system_prompt,
            print_debug=False,
            msg_history=msg_history,
            temperature=temperature,
        )
        review = extract_json_between_markers(llm_review)

    if num_reflections > 1:
        for j in range(num_reflections - 1):
            text, msg_history = get_response_from_llm(
                reviewer_reflection_prompt,
                client=client,
                model=model,
                system_message=reviewer_system_prompt,
                msg_history=msg_history,
                temperature=temperature,
            )
            review = extract_json_between_markers(text)
            assert review is not None, "Failed to extract JSON from LLM output"
            if "I am done" in text:
                break

    if return_msg_history:
        return review, msg_history
    else:
        return review


reviewer_reflection_prompt = """第 {current_round}/{num_reflections} 轮。
在你的思考中，首先仔细考虑你刚刚创建的审阅的准确性和可靠性。
纳入你认为对评估论文重要的任何其他因素。
确保审阅清晰简洁，且 JSON 格式正确。
不要让事情过于复杂。
在下一轮尝试中，尝试改进和完善你的审阅。
除非有明显的严重问题，否则应坚持原审阅的核心精神。

请使用与之前相同的格式回复：
THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

如果没有需要改进的地方，只需在思考之后完全重复之前的 JSON，并在思考结束但 JSON 之前注明 "I am done"。
仅在你不再做任何更改时才包含 "I am done"。"""


def load_paper(pdf_path, num_pages=None, min_size=100):
    try:
        if num_pages is None:
            text = pymupdf4llm.to_markdown(pdf_path)
        else:
            reader = PdfReader(pdf_path)
            min_pages = min(len(reader.pages), num_pages)
            text = pymupdf4llm.to_markdown(pdf_path, pages=list(range(min_pages)))
        if len(text) < min_size:
            raise Exception("Text too short")
    except Exception as e:
        print(f"Error with pymupdf4llm, falling back to pymupdf: {e}")
        try:
            doc = pymupdf.open(pdf_path)
            if num_pages:
                doc = doc[:num_pages]
            text = ""
            for page in doc:
                text += page.get_text()
            if len(text) < min_size:
                raise Exception("Text too short")
        except Exception as e:
            print(f"Error with pymupdf, falling back to pypdf: {e}")
            reader = PdfReader(pdf_path)
            if num_pages is None:
                pages = reader.pages
            else:
                pages = reader.pages[:num_pages]
            text = "".join(page.extract_text() for page in pages)
            if len(text) < min_size:
                raise Exception("Text too short")
    return text


def load_review(json_path):
    with open(json_path, "r") as json_file:
        loaded = json.load(json_file)
    return loaded["review"]


dir_path = os.path.dirname(os.path.realpath(__file__))

fewshot_papers = [
    os.path.join(dir_path, "fewshot_examples/132_automated_relational.pdf"),
    os.path.join(dir_path, "fewshot_examples/attention.pdf"),
    os.path.join(dir_path, "fewshot_examples/2_carpe_diem.pdf"),
]

fewshot_reviews = [
    os.path.join(dir_path, "fewshot_examples/132_automated_relational.json"),
    os.path.join(dir_path, "fewshot_examples/attention.json"),
    os.path.join(dir_path, "fewshot_examples/2_carpe_diem.json"),
]


def get_review_fewshot_examples(num_fs_examples=1):
    fewshot_prompt = """
以下是从之前的机器学习会议中摘录的一些审阅示例。
请注意，虽然每篇审阅根据审阅人的风格格式各异，但这些审阅结构良好，因此易于浏览。
"""
    for paper_path, review_path in zip(
        fewshot_papers[:num_fs_examples], fewshot_reviews[:num_fs_examples]
    ):
        txt_path = paper_path.replace(".pdf", ".txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                paper_text = f.read()
        else:
            paper_text = load_paper(paper_path)
        review_text = load_review(review_path)
        fewshot_prompt += f"""
Paper:

```
{paper_text}
```

Review:

```
{review_text}
```
"""
    return fewshot_prompt


meta_reviewer_system_prompt = """你是一个机器学习会议的领域主席。
你负责对一篇由 {reviewer_count} 位审阅人审阅过的论文进行元审阅。
你的工作是将这些审阅意见汇总为一份相同格式的元审阅。
请对你的决定持批判和谨慎态度，寻求共识，并尊重所有审阅人的意见。"""


def get_meta_review(model, client, temperature, reviews):
    review_text = ""
    for i, r in enumerate(reviews):
        review_text += f"""
Review {i + 1}/{len(reviews)}:
```
{json.dumps(r)}
```
"""
    base_prompt = neurips_form + review_text
    llm_review, _ = get_response_from_llm(
        base_prompt,
        model=model,
        client=client,
        system_message=meta_reviewer_system_prompt.format(reviewer_count=len(reviews)),
        print_debug=False,
        msg_history=None,
        temperature=temperature,
    )
    meta_review = extract_json_between_markers(llm_review)
    return meta_review
