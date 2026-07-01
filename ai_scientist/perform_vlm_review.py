import os
import hashlib
import pymupdf
import re
import base64
from ai_scientist.vlm import (
    get_response_from_vlm,
    get_batch_responses_from_vlm,
    extract_json_between_markers,
)

from ai_scientist.perform_llm_review import load_paper


def encode_image_to_base64(image_data):
    """Encode image data to base64 string."""
    if isinstance(image_data, str):
        with open(image_data, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    elif isinstance(image_data, list):
        return base64.b64encode(image_data[0]).decode("utf-8")
    elif isinstance(image_data, bytes):
        return base64.b64encode(image_data).decode("utf-8")
    else:
        raise TypeError(f"Unsupported image data type: {type(image_data)}")


reviewer_system_prompt_base = (
    "你是一位正在审阅提交给顶级机器学习会议的论文的AI研究员。"
    "请对你的决定持批判和谨慎态度。"
)

img_cap_ref_review_prompt = """论文摘要如下：

{abstract}

你将通过视觉 API 获得一张图像。作为一名严谨的科学家审阅人，你的任务是：
  1. 仔细检查提供的图像。
  2. 以科学的方式详细描述图像所展示的内容。
  3. 批判性地分析图像内容是否与给定的图注一致：

{caption}

  4. 我们还有正文中提到该图的引用：

{main_text_figrefs}

你应该：
  - 详细检查图：总结图中的元素（例如坐标轴名称）并描述展示了什么信息（例如损失曲线单调递减但在 X 个 epoch 后趋于平稳）
  - 对图本身提出任何潜在的改进或问题（例如，缺少图例、标签不清晰、没有有意义的结论、与图注声称的内容不匹配）。
  - 评价图注：它是否准确描述了图？太长/太短？是否包含了简洁的要点？
  - 审查正文引用（figrefs）对图的解释程度：是否缺失？是否充分描述了图的内容、背景或目的？

最后，请按照以下格式回复：

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```
在 <JSON> 中，按顺序以 JSON 格式提供审阅，包含以下字段：
- "Img_description": "<在此描述图的内容>"
- "Img_review": "<你对图本身的分析，包括任何改进建议>"
- "Caption_review": "<你对图注与图匹配程度的评估以及任何建议>"
- "Figrefs_review": "<你对正文引用是否充分描述或整合了该图的看法>"

在 <THOUGHT> 中，首先全面推理你的观察、对齐分析以及任何建议的改进。可以写得很长。
然后在 <JSON> 中提供最终的结构化输出。
确保 JSON 有效且格式正确，因为它将被自动解析。"""


img_cap_selection_prompt = """论文摘要如下：

{abstract}

你将通过视觉 API 获得一张图像。作为一名严谨的科学家审阅人，你的任务是：
  1. 仔细检查提供的图像。
  2. 以科学的方式详细描述图像所展示的内容。
  3. 批判性地分析图像内容是否与给定的图注一致：

{caption}

  4. 我们还有正文中提到该图的引用：

{main_text_figrefs}

  5. 我们展示内容的页面有限：

{reflection_page_info}

你应该：
  - 详细检查图：总结图中的元素（例如坐标轴名称）并描述展示了什么信息（例如损失曲线单调递减但在 X 个 epoch 后趋于平稳）
  - 评价图注：它是否准确描述了图？太长/太短？是否包含了简洁的要点？
  - 审查正文引用（figrefs）对图的解释程度：是否缺失？是否充分描述了图的内容、背景或目的？

在考虑以上所有因素后，你应该仔细评估：
  - 考虑到当前的页数限制，此图像及其相关文本是否为论文的科学论证增加了显著价值？
  - 考虑到当前的页数限制，此图像是否信息过于稀疏？是否应与其他正文中的图合并？
  - 此图是否包含子图？
  - 此图是否信息量不足？例如，一些图可能显示高度非常相似、难以区分的条形，或者以无法有效传达有意义差异或模式的方式呈现数据。

最后，请按照以下格式回复：

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```
在 <JSON> 中，按顺序以 JSON 格式提供审阅，包含以下字段：
- "Img_description": "<在此描述图的内容>"
- "Img_review": "<你对图本身的分析，包括任何改进建议>"
- "Caption_review": "<你对图注与图匹配程度的评估以及任何建议>"
- "Figrefs_review": "<你对正文引用是否充分描述或整合了该图的看法>"
- "Overall_comments": "<你对此图是否为论文增加了显著价值的看法。是否应该移至附录？>"
- "Containing_sub_figures": "<此图是否包含多个子图？你认为此图中的信息是否密集？如果不密集，是否建议将其与正文中的其他图合并？如果包含子图，它们的大小和位置是否良好对齐？如果没有，请描述问题。>"
- "Informative_review": "<此图是否信息丰富？它是否有效传达了有意义的差异或模式？还是以难以区分差异的方式显示数据（例如条形高度非常相似）？>"

在 <THOUGHT> 中，首先全面推理你的观察、对齐分析以及任何建议的改进。可以写得很长。
然后在 <JSON> 中提供最终的结构化输出。
确保 JSON 有效且格式正确，因为它将被自动解析。"""

img_review_prompt = """

你将通过视觉 API 获得一张图像。作为一名严谨的科学家审阅人，你的任务是：
  1. 仔细检查提供的图像。
  2. 以科学的方式详细描述图像所展示的内容。

你应该：
  - 详细检查图：总结图中的元素（例如坐标轴名称）并描述展示了什么信息（例如损失曲线单调递减但在 X 个 epoch 后趋于平稳）
  - 对图本身提出任何潜在的改进或问题（例如，缺少图例、标签不清晰、没有有意义的结论、与图注声称的内容不匹配）。

最后，请按照以下格式回复：

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```
在 <JSON> 中，按顺序以 JSON 格式提供审阅，包含以下字段：
- "Img_description": "<在此描述图的内容>"
- "Img_review": "<你对图本身的分析，包括任何改进建议>"

在 <THOUGHT> 中，首先全面推理你的观察、对齐分析以及任何建议的改进。可以写得很长。
然后在 <JSON> 中提供最终的结构化输出。
确保 JSON 有效且格式正确，因为它将被自动解析。"""


def extract_figure_screenshots(
    pdf_path,
    img_folder_path,
    num_pages=None,
    min_text_length=50,
    min_vertical_gap=30,
):
    """
    Extract screenshots for figure captions ("Figure X." or "Figure X:")
    and also gather text blocks (anywhere in the PDF) mentioning that
    exact figure with "Figure", "Fig.", or "Fig-ure" (including line breaks).
    Avoid partial matches, e.g. "Figure 11" doesn't match "Figure 1".
    """
    os.makedirs(img_folder_path, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    page_range = (
        range(len(doc)) if num_pages is None else range(min(num_pages, len(doc)))
    )

    # ---------- (A) EXTRACT ALL TEXT BLOCKS FROM THE DOCUMENT ----------
    text_blocks = []  # will hold dicts: { 'page': int, 'bbox': Rect, 'text': str }
    for page_num in page_range:
        page = doc[page_num]
        try:
            blocks = page.get_text("blocks")
            # blocks: [x0, y0, x1, y1, text, block_no, ...]
            for b in blocks:
                txt = b[4].strip()
                if txt:
                    bbox = pymupdf.Rect(b[0], b[1], b[2], b[3])
                    text_blocks.append({"page": page_num, "bbox": bbox, "text": txt})
        except Exception as e:
            print(f"Error extracting text from page {page_num}: {e}")

    # ---------- (B) REGEX FOR FIGURE CAPTIONS  ----------
    # Captures the figure label so we can reference it later (group name 'fig_label').
    # Example matches: "Figure 1:", "Figure (A).2.", "Figure A.1:"
    figure_caption_pattern = re.compile(
        r"^(?:Figure)\s+(?P<fig_label>"
        r"(?:\d+"  # "1", "11", ...
        r"|[A-Za-z]+\.\d+"  # "A.1", "S2.3"
        r"|\(\s*[A-Za-z]+\s*\)\.\d+"  # "(A).2"
        r")"
        r")(?:\.|:)",  # Must end with "." or ":"
        re.IGNORECASE,
    )

    # ---------- (C) DETECT SUB-FIGURE CAPTIONS (e.g. "(a)")  ----------
    subfigure_pattern = re.compile(r"\(\s*[a-zA-Z]\s*\)")

    def is_subfigure_caption(txt):
        return bool(subfigure_pattern.search(txt))

    # ---------- (D) MAIN ROUTINE: LOOP OVER PAGES AND CAPTIONS ----------
    result_pairs = []

    for page_num in page_range:
        page = doc[page_num]
        page_rect = page.rect

        # All text blocks for this page
        page_blocks = [b for b in text_blocks if b["page"] == page_num]
        # Sort top-to-bottom
        page_blocks.sort(key=lambda b: b["bbox"].y0)

        # ----- (D.1) Find figure captions -----
        for blk in page_blocks:
            caption_text = blk["text"]
            m = figure_caption_pattern.match(caption_text)
            if not m:
                continue  # not a figure caption

            fig_label = m.group("fig_label")  # e.g. "1", "A.1", "(A).2", etc.
            fig_x0, fig_y0, fig_x1, fig_y1 = blk["bbox"]

            # (a) Find a large text block above the caption (on the same page)
            above_blocks = []
            for ab in page_blocks:
                if ab["bbox"].y1 < fig_y0:
                    # vertical gap
                    ab_height_gap = fig_y0 - ab["bbox"].y1
                    # horizontal overlap
                    overlap_x = min(fig_x1, ab["bbox"].x1) - max(fig_x0, ab["bbox"].x0)
                    width_min = min((fig_x1 - fig_x0), (ab["bbox"].x1 - ab["bbox"].x0))
                    horiz_overlap_ratio = (
                        overlap_x / float(width_min) if width_min > 0 else 0.0
                    )

                    if (
                        len(ab["text"]) >= min_text_length
                        and not is_subfigure_caption(ab["text"])
                        and ab_height_gap >= min_vertical_gap
                        and horiz_overlap_ratio > 0.3
                    ):
                        above_blocks.append(ab)

            # pick the block with the largest bottom edge
            if above_blocks:
                above_block = max(above_blocks, key=lambda b: b["bbox"].y1)
                clip_top = above_block["bbox"].y1
            else:
                clip_top = page_rect.y0

            clip_left = fig_x0
            clip_right = fig_x1
            clip_bottom = fig_y0

            # (b) Create figure screenshot
            if (clip_bottom > clip_top) and (clip_right > clip_left):
                clip_rect = pymupdf.Rect(clip_left, clip_top, clip_right, clip_bottom)
                pix = page.get_pixmap(clip=clip_rect, dpi=150)

                fig_label_escaped = re.escape(fig_label)
                # unique filename
                fig_hash = hashlib.md5(
                    f"figure_{fig_label_escaped}_{page_num}_{clip_rect}".encode()
                ).hexdigest()[:10]
                fig_filename = (
                    f"figure_{fig_label_escaped}_Page_{page_num+1}_{fig_hash}.png"
                )
                fig_filepath = os.path.join(img_folder_path, fig_filename)
                pix.save(fig_filepath)

                # (c) Now find references across the ENTIRE DOCUMENT
                #     We'll build a pattern that matches:
                #         Figure/Fig./Fig-ure + possible line break + fig_label
                #     We also ensure we do NOT match if there's a digit/letter
                #     immediately after fig_label (so "Figure 11" won't match "Figure 1").
                fig_label_escaped = re.escape(fig_label)
                # negative lookahead (?![0-9A-Za-z]) ensures no letter/digit follows
                main_text_figure_pattern = re.compile(
                    rf"(?:Fig(?:\.|-\s*ure)?|Figure)\s*{fig_label_escaped}(?![0-9A-Za-z])",
                    re.IGNORECASE,
                )

                references_in_doc = []
                for tb in text_blocks:
                    # exclude the caption block itself
                    if tb is blk:
                        continue
                    # see if it references this figure label
                    if main_text_figure_pattern.search(tb["text"]):
                        references_in_doc.append(tb["text"])

                # (d) Create the final result item
                result_pairs.append(
                    {
                        "img_name": f"figure_{fig_label_escaped}",
                        "caption": caption_text,
                        "images": [fig_filepath],
                        "main_text_figrefs": references_in_doc,
                    }
                )

    return result_pairs


def extract_abstract(text):
    # Split text into lines
    lines = text.split("\n")

    # Regex to identify a heading line: starts with # after optional spaces
    # e.g. "### Some Heading"
    heading_pattern = re.compile(r"^\s*#+\s*(.*)$")

    # Find the line containing "abstract" in a heading
    abstract_start = None
    for i, line in enumerate(lines):
        # Check if this line is a heading
        match = heading_pattern.match(line)
        if match:
            # Extract the heading text after '#'
            heading_text = match.group(1)
            if "abstract" in heading_text.lower():
                abstract_start = i
                break

    if abstract_start is None:
        # No abstract heading found
        return ""

    # From abstract_start, collect lines until the next heading
    abstract_lines = []
    for j in range(abstract_start + 1, len(lines)):
        # Check if this line is another heading
        if heading_pattern.match(lines[j]):
            # We've hit the next section heading, stop extraction
            break
        # Otherwise, accumulate the line as part of the abstract
        abstract_lines.append(lines[j])

    # Join the abstract lines into a single string
    abstract_text = "\n".join(abstract_lines).strip()
    return abstract_text


def generate_vlm_img_cap_ref_review(img, abstract, model, client):
    prompt = img_cap_ref_review_prompt.format(
        abstract=abstract,
        caption=img["caption"],
        main_text_figrefs=img["main_text_figrefs"],
    )
    content, _ = get_response_from_vlm(
        prompt, img["images"], client, model, reviewer_system_prompt_base
    )
    img_cap_ref_review_json = extract_json_between_markers(content)
    return img_cap_ref_review_json


def generate_vlm_img_review(img, model, client):
    prompt = img_review_prompt
    content, _ = get_response_from_vlm(
        prompt, img["images"], client, model, reviewer_system_prompt_base
    )
    img_review_json = extract_json_between_markers(content)
    return img_review_json


def perform_imgs_cap_ref_review(client, client_model, pdf_path):
    paper_txt = load_paper(pdf_path)
    img_folder_path = os.path.join(
        os.path.dirname(pdf_path),
        f"{os.path.splitext(os.path.basename(pdf_path))[0]}_imgs",
    )
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)
    img_pairs = extract_figure_screenshots(pdf_path, img_folder_path)
    img_reviews = {}
    abstract = extract_abstract(paper_txt)
    for img in img_pairs:
        review = generate_vlm_img_cap_ref_review(img, abstract, client_model, client)
        img_reviews[img["img_name"]] = review
    return img_reviews


def detect_duplicate_figures(client, client_model, pdf_path):
    paper_txt = load_paper(pdf_path)
    img_folder_path = os.path.join(
        os.path.dirname(pdf_path),
        f"{os.path.splitext(os.path.basename(pdf_path))[0]}_imgs",
    )
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)
    img_pairs = extract_figure_screenshots(pdf_path, img_folder_path)

    messages = [
        {
            "role": "system",
            "content": (
                "你是一位识别重复或高度相似图像的专家。"
                "请分析这些图像并判断它们是重复的还是同一可视化的变体。"
                "回复格式：推理，然后在 `重复图: <重复图名称列表>` 中列出。"
                "确保使用论文中出现的准确图名称（例如 Figure 1、Figure 2b 等）。"
                "如果没有发现重复，请回复 `未发现重复`。"
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "这些图像中是否有重复或高度相似的？如果有，请指出哪些是相似的并解释原因。关注内容相似性，而不仅仅是视觉风格。",
                }
            ],
        },
    ]

    # Add images in the correct format
    for img_info in img_pairs:
        messages[1]["content"].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image_to_base64(img_info['images'][0])}"
                },
            }
        )

    try:
        response = client.chat.completions.create(
            model=client_model,
            messages=messages,
            max_tokens=1000,
        )

        analysis = response.choices[0].message.content

        return analysis

    except Exception as e:
        print(f"Error analyzing images: {e}")
        return {"error": str(e)}


def generate_vlm_img_selection_review(
    img, abstract, model, client, reflection_page_info
):
    prompt = img_cap_selection_prompt.format(
        abstract=abstract,
        caption=img["caption"],
        main_text_figrefs=img["main_text_figrefs"],
        reflection_page_info=reflection_page_info,
    )
    content, _ = get_response_from_vlm(
        prompt, img["images"], client, model, reviewer_system_prompt_base
    )
    img_cap_ref_review_json = extract_json_between_markers(content)
    return img_cap_ref_review_json


def perform_imgs_cap_ref_review_selection(
    client, client_model, pdf_path, reflection_page_info
):
    paper_txt = load_paper(pdf_path)
    img_folder_path = os.path.join(
        os.path.dirname(pdf_path),
        f"{os.path.splitext(os.path.basename(pdf_path))[0]}_imgs",
    )
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)
    img_pairs = extract_figure_screenshots(pdf_path, img_folder_path)
    img_reviews = {}
    abstract = extract_abstract(paper_txt)
    for img in img_pairs:
        review = generate_vlm_img_selection_review(
            img, abstract, client_model, client, reflection_page_info
        )
        img_reviews[img["img_name"]] = review
    return img_reviews
