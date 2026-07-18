import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agent import CourseAssistant
from config import Settings
from rag import RAGService


def settings_for(tmp_path: Path) -> Settings:
    return Settings(
        api_key="test",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        score_threshold=0.0,
        top_k=3,
    )


def test_grounded_answer_and_multiturn_history(tmp_path, hash_embeddings):
    material = tmp_path / "讲义.md"
    material.write_text("# 列表\n列表是可变、有序序列，append 可以添加元素。", encoding="utf-8")
    settings = settings_for(tmp_path)
    rag = RAGService(settings, embeddings=hash_embeddings)
    rag.add_materials([material])
    answer_json = json.dumps(
        {
            "found": True,
            "direct_answer": "列表是可变序列。",
            "explanation": "可以使用 append 添加元素。",
            "citations": [
                {"source": "讲义.md", "location": "列表", "quote": "列表是可变、有序序列"}
            ],
            "study_advice": ["练习 append"],
            "confidence": "高",
        },
        ensure_ascii=False,
    )
    model = FakeListChatModel(responses=[answer_json, "列表如何添加元素", answer_json])
    assistant = CourseAssistant(settings, rag, model=model)
    first = assistant.ask("列表能修改吗？", session_id="student-1")
    second = assistant.ask("那怎么添加呢？", session_id="student-1")
    assert first["result"].found
    assert "资料依据" in first["markdown"]
    assert second["standalone_query"] == "列表如何添加元素"
    assert len(assistant._history("student-1").messages) == 4


def test_no_material_returns_fixed_refusal(tmp_path, hash_embeddings):
    settings = settings_for(tmp_path)
    rag = RAGService(settings, embeddings=hash_embeddings)
    assistant = CourseAssistant(settings, rag, model=FakeListChatModel(responses=[]))
    result = assistant.ask("量子力学是什么？", session_id="empty")
    assert result["result"].found is False
    assert "未找到资料依据" in result["markdown"]
    assert "不能基于资料回答" in result["markdown"]
    assert "导入包含该知识点的资料" in result["markdown"]


def test_outline_rejects_citation_not_found_in_retrieved_material(
    tmp_path, hash_embeddings
):
    material = tmp_path / "讲义.md"
    material.write_text(
        "# 第一章 Python 基础\nPython 程序由解释器执行。",
        encoding="utf-8",
    )
    settings = settings_for(tmp_path)
    rag = RAGService(settings, embeddings=hash_embeddings)
    rag.add_materials([material])
    outline_json = json.dumps(
        {
            "title": "第一章复习提纲",
            "overview": "Python 基础",
            "sections": [
                {
                    "title": "运行方式",
                    "key_points": ["Python 程序由解释器执行"],
                    "common_mistakes": [],
                }
            ],
            "review_checklist": ["能够说明 Python 程序如何执行"],
            "citations": [
                {
                    "source": "讲义.md",
                    "location": "第一章",
                    "quote": "这段文字并不存在于资料中",
                }
            ],
        },
        ensure_ascii=False,
    )
    assistant = CourseAssistant(
        settings,
        rag,
        model=FakeListChatModel(responses=[outline_json]),
    )

    with pytest.raises(ValueError, match="一个章节、一个检查项和一个有效资料引用"):
        assistant.generate_outline("第一章 Python 基础")


def test_model_cannot_force_an_unsupported_answer(tmp_path, hash_embeddings):
    material = tmp_path / "讲义.md"
    material.write_text("# 列表\n列表是可变序列。", encoding="utf-8")
    settings = settings_for(tmp_path)
    rag = RAGService(settings, embeddings=hash_embeddings)
    rag.add_materials([material])
    unsupported = json.dumps(
        {
            "found": True,
            "direct_answer": "我是某个特定模型。",
            "explanation": "",
            "citations": [
                {"source": "讲义.md", "location": "列表", "quote": "我是某个特定模型"}
            ],
            "study_advice": [],
            "confidence": "高",
        },
        ensure_ascii=False,
    )
    assistant = CourseAssistant(
        settings, rag, model=FakeListChatModel(responses=[unsupported])
    )

    result = assistant.ask("你是什么模型？", session_id="unsupported")

    assert result["result"].found is False
    assert "未找到资料依据" in result["markdown"]
    assert "不能基于资料回答" in result["markdown"]


def test_supported_answer_gets_fallback_citation(tmp_path, hash_embeddings):
    material = tmp_path / "讲义.md"
    material.write_text(
        "# 组合数据类型\n列表是有序、可变序列。元组是有序、不可变序列。",
        encoding="utf-8",
    )
    settings = settings_for(tmp_path)
    rag = RAGService(settings, embeddings=hash_embeddings)
    rag.add_materials([material])
    answer_without_citation = json.dumps(
        {
            "found": True,
            "direct_answer": "列表可修改，元组不可修改。",
            "explanation": "列表是可变序列，元组是不可变序列。",
            "citations": [],
            "study_advice": [],
            "confidence": "高",
        },
        ensure_ascii=False,
    )
    assistant = CourseAssistant(
        settings, rag, model=FakeListChatModel(responses=[answer_without_citation])
    )

    result = assistant.ask("列表和元组有什么区别？", session_id="fallback-citation")

    assert result["result"].found is True
    assert result["result"].citations[0].source == "讲义.md"
    assert result["result"].study_advice
    assert "资料依据" in result["markdown"]
    assert "学习建议" in result["markdown"]


def test_quiz_missing_citation_gets_fallback(tmp_path, hash_embeddings):
    material = tmp_path / "讲义.md"
    material.write_text("# 列表\n列表是有序、可变序列，可以使用 append 添加元素。", encoding="utf-8")
    settings = settings_for(tmp_path)
    rag = RAGService(settings, embeddings=hash_embeddings)
    rag.add_materials([material])
    quiz_without_citation = json.dumps(
        {
            "title": "列表练习",
            "difficulty": "基础",
            "questions": [
                {
                    "number": 1,
                    "question_type": "简答题",
                    "question": "列表是否可变？",
                    "options": [],
                    "answer": "列表是可变序列。",
                    "explanation": "资料说明列表是有序、可变序列。",
                }
            ],
        },
        ensure_ascii=False,
    )
    assistant = CourseAssistant(
        settings, rag, model=FakeListChatModel(responses=[quiz_without_citation])
    )

    result = assistant.generate_quiz("列表", count=1)

    question = result["result"].questions[0]
    assert question.citation.source == "讲义.md"
    assert "依据" in result["markdown"]
