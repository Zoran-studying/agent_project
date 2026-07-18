"""Validated data models and Markdown renderers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    source: str = Field(description="资料文件名")
    location: str = Field(description="页码、章节或片段位置")
    quote: str = Field(description="支持结论的简短原文")


class AnswerResult(BaseModel):
    found: bool = True
    direct_answer: str
    explanation: str = ""
    citations: list[Citation] = Field(default_factory=list)
    study_advice: list[str] = Field(default_factory=list)
    confidence: Literal["高", "中", "低"] = "中"

    def to_markdown(self) -> str:
        if not self.found:
            return "\n".join(
                [
                    "## 未找到资料依据",
                    "",
                    "当前知识库中没有检索到足够可靠的资料依据，因此不能基于资料回答这个问题。",
                    "",
                    "你可以尝试：",
                    "",
                    "- 换用课程资料中的章节名、概念名或关键词重新提问",
                    "- 先导入包含该知识点的资料，并重建向量索引",
                    "- 如果确认资料中存在相关内容，可适当降低相似度阈值后再试",
                ]
            )
        lines = [
            "## 直接答案",
            "",
            self.direct_answer,
        ]
        if self.explanation.strip():
            lines.extend(["", "## 知识点解释", "", self.explanation])
        lines.extend(["", "## 资料依据", ""])
        lines.extend(
            f"- **{c.source} · {c.location}**：{c.quote}" for c in self.citations
        )
        if not self.citations:
            lines.append("- 无可用引用")
        lines.extend(["", "## 学习建议", ""])
        advice = self.study_advice or ["结合资料依据复述答案，并用一个例子检查自己是否理解。"]
        lines.extend(f"- {item}" for item in advice)
        lines.extend(["", f"**置信度：{self.confidence}**"])
        return "\n".join(lines)


class OutlineSection(BaseModel):
    title: str
    key_points: list[str] = Field(min_length=1)
    common_mistakes: list[str] = Field(default_factory=list)


class OutlineResult(BaseModel):
    title: str
    overview: str
    sections: list[OutlineSection] = Field(min_length=1)
    review_checklist: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", "", self.overview, ""]
        for section in self.sections:
            lines.extend([f"## {section.title}", ""])
            lines.extend(f"- {item}" for item in section.key_points)
            if section.common_mistakes:
                lines.extend(["", "**易错点**", ""])
                lines.extend(f"- {item}" for item in section.common_mistakes)
            lines.append("")
        lines.extend(["## 复习检查清单", ""])
        lines.extend(f"- [ ] {item}" for item in self.review_checklist)
        lines.extend(["", "## 资料依据", ""])
        lines.extend(
            f"- **{c.source} · {c.location}**：{c.quote}" for c in self.citations
        )
        return "\n".join(lines)


class QuizQuestion(BaseModel):
    number: int
    question_type: str
    question: str
    options: list[str] = Field(default_factory=list)
    answer: str
    explanation: str
    citation: Citation = Field(
        default_factory=lambda: Citation(source="", location="", quote="")
    )


class QuizResult(BaseModel):
    title: str
    difficulty: str
    questions: list[QuizQuestion]

    def to_markdown(self, show_answers: bool = True) -> str:
        lines = [f"# {self.title}", "", f"**难度：{self.difficulty}**", ""]
        for q in self.questions:
            lines.extend([f"## {q.number}. [{q.question_type}] {q.question}", ""])
            lines.extend(f"- {option}" for option in q.options)
            if show_answers:
                lines.extend(
                    [
                        "",
                        f"**参考答案：** {q.answer}",
                        "",
                        f"**解析：** {q.explanation}",
                        "",
                        f"**依据：** {q.citation.source} · {q.citation.location} — {q.citation.quote}",
                    ]
                )
            lines.append("")
        return "\n".join(lines)


class RetrievedChunk(BaseModel):
    content: str
    source: str
    location: str
    score: float
    chapter: str = ""
    chunk_id: str = ""

    def display_label(self) -> str:
        return f"{self.source} · {self.location} · 相似度 {self.score:.3f}"
