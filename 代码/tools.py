"""LangChain tools exposed by the course assistant."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


_rag_service: Any = None
_agent_service: Any = None


def configure_services(rag_service: Any, agent_service: Any = None) -> None:
    global _rag_service, _agent_service
    _rag_service = rag_service
    _agent_service = agent_service


def _require_rag() -> Any:
    if _rag_service is None:
        raise RuntimeError("RAG 服务尚未初始化")
    return _rag_service


@tool
def load_course_materials(paths: list[str]) -> str:
    """读取并索引一组课程资料文件，支持 txt、md、pdf 和 docx。"""
    return json.dumps(_require_rag().add_materials(paths), ensure_ascii=False)


@tool
def search_course_content(query: str, top_k: int = 5) -> str:
    """根据学生问题检索课程资料，返回相似度最高的资料片段和来源。"""
    hits = _require_rag().search(query, top_k=top_k)
    return json.dumps([item.model_dump() for item in hits], ensure_ascii=False)


@tool
def summarize_chapter(chapter: str, source: str = "") -> str:
    """根据已索引资料为指定章节生成结构化复习提纲。"""
    if _agent_service is None:
        raise RuntimeError("问答服务尚未初始化")
    payload = _agent_service.generate_outline(chapter=chapter, source=source or None)
    return payload["markdown"]


@tool
def generate_quiz(
    chapter: str,
    count: int = 5,
    difficulty: str = "中等",
    question_types: str = "选择题、简答题",
) -> str:
    """依据课程资料生成练习题、参考答案、解析和来源。"""
    if _agent_service is None:
        raise RuntimeError("问答服务尚未初始化")
    payload = _agent_service.generate_quiz(
        chapter=chapter,
        count=count,
        difficulty=difficulty,
        question_types=question_types,
    )
    return payload["markdown"]


@tool
def save_result(filename: str, content: str) -> str:
    """将结构化结果安全保存到项目 outputs 目录。"""
    rag = _require_rag()
    safe_name = Path(filename).name
    if Path(safe_name).suffix.lower() not in {".md", ".json"}:
        safe_name += ".md"
    target = rag.settings.output_dir / safe_name
    target.write_text(content, encoding="utf-8")
    return f"文件已保存：{target}"


ALL_TOOLS = [
    load_course_materials,
    search_course_content,
    summarize_chapter,
    generate_quiz,
    save_result,
]

