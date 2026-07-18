"""Prompt templates used by the LangChain LCEL pipelines."""

from langchain_core.prompts import ChatPromptTemplate


REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你负责把多轮对话中的当前问题改写成可独立检索的问题。"
            "不得回答问题，只输出改写后的单行查询；若问题已经完整则原样返回。",
        ),
        (
            "human",
            "对话历史：\n{history}\n\n当前问题：{question}\n\n独立检索查询：",
        ),
    ]
)


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是严谨的课程资料问答助手。只能使用给定资料片段作答，不得使用外部知识补全。"
            "先判断资料片段是否足以直接回答问题。若资料没有明确答案、仅部分相关、需要猜测，"
            "必须令 found=false、direct_answer='当前资料中未找到相关信息。'，其余说明、引用和建议均留空；"
            "禁止根据常识、模型身份或对话暗示补全。若资料足以回答，令 found=true，且每条引用的 quote "
            "必须逐字摘自给定片段。direct_answer 直接回答问题；当问题涉及概念、区别、步骤、原因或易混点时，"
            "explanation 用 1-3 句解释资料中的相关知识点，否则可留空。study_advice 必须给 1-3 条基于资料的"
            "可执行学习建议。回答必须简洁、准确。只输出符合下方格式要求的 JSON，"
            "不要使用 Markdown 代码围栏。\n{format_instructions}",
        ),
        (
            "human",
            "对话历史：\n{history}\n\n问题：{question}\n\n资料片段：\n{context}",
        ),
    ]
)


OUTLINE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是课程复习教练，只能依据给定资料生成分层复习提纲。覆盖核心概念、关键步骤、"
            "易错点和检查清单。硬性要求：sections 至少包含一个有知识点的章节，"
            "review_checklist 至少包含一个非空检查项，citations 至少包含一个有效引用；"
            "每条引用的 source 必须来自给定资料，quote 必须逐字摘自资料片段。"
            "任何必填内容都不得使用空字符串或空列表。只输出合法 JSON，不要代码围栏。\n"
            "{format_instructions}",
        ),
        (
            "human",
            "资料范围：{scope}\n目标章节：{chapter}\n\n资料片段：\n{context}",
        ),
    ]
)


QUIZ_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是课程出题教师。只能依据给定资料生成题目、答案和解析，答案不得超出资料。"
            "严格生成指定数量，题号从 1 连续递增。只输出合法 JSON，不要代码围栏。\n"
            "{format_instructions}",
        ),
        (
            "human",
            "资料范围：{scope}\n章节：{chapter}\n题量：{count}\n难度：{difficulty}\n"
            "题型：{question_types}\n\n资料片段：\n{context}",
        ),
    ]
)
