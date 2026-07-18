"""Run the real retrieval/model acceptance suite and write a Markdown report."""

from __future__ import annotations

import json
from datetime import datetime

from agent import CourseAssistant
from config import Settings
from rag import RAGService
from tools import configure_services


def main() -> None:
    settings = Settings()
    settings.validate_api()
    rag = RAGService(settings)
    samples = [
        settings.data_dir / "Python程序设计课程讲义.md",
        settings.data_dir / "Python实验指导书.md",
    ]
    rag.add_materials(samples)
    assistant = CourseAssistant(settings, rag)
    configure_services(rag, assistant)

    questions = [
        "Python 中列表和元组有什么区别？",
        "break 和 continue 分别有什么作用？",
        "文件操作为什么推荐使用 with open？",
        "函数没有显式 return 时会返回什么？",
        "字典的键有什么要求？",
    ]
    reports = []
    for index, question in enumerate(questions, 1):
        session_id = f"acceptance-{index}"
        assistant.clear_history(session_id)
        payload = assistant.ask(question, session_id=session_id)
        reports.append(f"# 问题 {index}\n\n**{question}**\n\n{payload['markdown']}")

    # Verify follow-up rewriting in a dedicated two-turn conversation.
    assistant.clear_history("multiturn")
    assistant.ask("列表是什么？", session_id="multiturn")
    followup = assistant.ask("那怎么添加元素？", session_id="multiturn")
    reports.append(
        "# 多轮追问验证\n\n"
        f"**原追问：** 那怎么添加元素？  \n"
        f"**改写查询：** {followup['standalone_query']}\n\n{followup['markdown']}"
    )

    assistant.clear_history("outside")
    refusal = assistant.ask("量子纠缠有哪些实验验证？", session_id="outside")
    if refusal["result"].found:
        raise AssertionError("资料外问题未触发拒答保护")
    reports.append(f"# 资料外问题验证\n\n{refusal['markdown']}")

    outline = assistant.generate_outline("第五章 异常与文件")
    quiz = assistant.generate_quiz("Python 基础", count=5, difficulty="中等")
    if len(quiz["result"].questions) != 5:
        raise AssertionError("模型未生成指定数量的练习题")
    (settings.output_dir / "真实API问答验收.md").write_text(
        "\n\n---\n\n".join(reports), encoding="utf-8"
    )
    (settings.output_dir / "真实API章节提纲.md").write_text(
        outline["markdown"], encoding="utf-8"
    )
    (settings.output_dir / "真实API练习题与答案.md").write_text(
        quiz["markdown"], encoding="utf-8"
    )
    summary = {
        "verified_at": datetime.now().isoformat(timespec="seconds"),
        "model": settings.model,
        "materials": [item["name"] for item in rag.list_materials()],
        "questions": len(questions),
        "followup_query": followup["standalone_query"],
        "outside_found": refusal["result"].found,
        "outline_sections": len(outline["result"].sections),
        "quiz_questions": len(quiz["result"].questions),
    }
    (settings.output_dir / "验收摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
