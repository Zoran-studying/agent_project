"""LangChain LCEL course assistant with grounded RAG generation."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

import httpx
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from config import Settings
from models import AnswerResult, Citation, OutlineResult, QuizResult, RetrievedChunk
from prompts import ANSWER_PROMPT, OUTLINE_PROMPT, QUIZ_PROMPT, REWRITE_PROMPT
from rag import RAGService


T = TypeVar("T", bound=BaseModel)

_MAX_INPUT_LENGTH = 500


class CourseAssistant:
    """High-level LangChain conversation and generation service."""

    def __init__(
        self,
        settings: Settings,
        rag: RAGService,
        model: BaseChatModel | None = None,
    ):
        self.settings = settings
        self.rag = rag
        if model is None:
            settings.validate_api()
            http_client = httpx.Client(verify=settings.verify_ssl, timeout=120)
            model = ChatOpenAI(
                api_key=settings.api_key,
                base_url=settings.base_url,
                model=settings.model,
                temperature=0.2,
                max_tokens=settings.max_tokens,
                timeout=120,
                max_retries=2,
                http_client=http_client,
                extra_body={"enable_thinking": settings.enable_thinking},
            )
        self.model = model
        string_parser = StrOutputParser()
        self.rewrite_chain = REWRITE_PROMPT | self.model | string_parser
        self.answer_parser = PydanticOutputParser(pydantic_object=AnswerResult)
        self.outline_parser = PydanticOutputParser(pydantic_object=OutlineResult)
        self.quiz_parser = PydanticOutputParser(pydantic_object=QuizResult)
        self.answer_chain = ANSWER_PROMPT | self.model | string_parser
        self.outline_chain = OUTLINE_PROMPT | self.model | string_parser
        self.quiz_chain = QUIZ_PROMPT | self.model | string_parser
        self.workflow = RunnableBranch(
            (
                lambda x: x.get("mode") == "outline",
                RunnableLambda(lambda x: self.generate_outline(**x["params"])),
            ),
            (
                lambda x: x.get("mode") == "quiz",
                RunnableLambda(lambda x: self.generate_quiz(**x["params"])),
            ),
            RunnableLambda(lambda x: self.ask(**x["params"])),
        )

    @staticmethod
    def _sanitize_text(text: str, max_length: int = _MAX_INPUT_LENGTH) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text[:max_length]

    def _history(self, session_id: str) -> SQLChatMessageHistory:
        return SQLChatMessageHistory(
            session_id=session_id,
            connection=f"sqlite:///{self.settings.history_db.as_posix()}",
        )

    @staticmethod
    def _history_text(history: SQLChatMessageHistory, limit: int = 8) -> str:
        lines = []
        for message in history.messages[-limit:]:
            role = "学生" if isinstance(message, HumanMessage) else "助手"
            lines.append(f"{role}：{message.content}")
        return "\n".join(lines) or "无"

    @staticmethod
    def _context(hits: list[RetrievedChunk]) -> str:
        blocks = []
        for index, hit in enumerate(hits, 1):
            blocks.append(
                f"[片段{index}] 来源={hit.source}; 位置={hit.location}; "
                f"相似度={hit.score:.3f}\n{hit.content}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _extract_json(raw: str) -> str:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        start, end = text.find("{"), text.rfind("}")
        return text[start : end + 1] if start >= 0 and end > start else text

    def _parse(self, raw: str, parser: PydanticOutputParser, schema: type[T]) -> T:
        cleaned = self._extract_json(raw)
        try:
            return parser.parse(cleaned)
        except (ValidationError, json.JSONDecodeError):
            repair_prompt = (
                "下面内容未满足 JSON 结构。请只修复格式，不添加事实，并严格返回合法 JSON。\n"
                f"目标结构：{parser.get_format_instructions()}\n原内容：{cleaned}"
            )
            repaired = self.model.invoke(repair_prompt).content
            try:
                result = schema.model_validate_json(self._extract_json(str(repaired)))
            except (ValidationError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"模型结构化输出解析失败（已尝试自动修复）：{exc}"
                ) from exc
            if schema.__name__ == "AnswerResult" and hasattr(result, "found"):
                try:
                    original = schema.model_validate_json(cleaned)
                    result.found = original.found
                except (ValidationError, json.JSONDecodeError):
                    pass
            return result

    @staticmethod
    def _ensure_citations(result: Any, hits: list[RetrievedChunk]) -> None:
        """Keep only citations whose quoted text actually occurs in a retrieved chunk."""
        contents_by_source: dict[str, list[str]] = {}
        for hit in hits:
            contents_by_source.setdefault(hit.source, []).append(
                re.sub(r"\s+", "", hit.content)
            )
        citations = []
        for citation in getattr(result, "citations", []):
            quote = re.sub(r"\s+", "", citation.quote)
            if quote and any(
                quote in content for content in contents_by_source.get(citation.source, [])
            ):
                citations.append(citation)
        if hasattr(result, "citations"):
            result.citations = citations

    @staticmethod
    def _fallback_citation(hits: list[RetrievedChunk]) -> Citation | None:
        if not hits:
            return None
        hit = hits[0]
        quote = re.sub(r"\s+", " ", hit.content).strip()[:120]
        return Citation(source=hit.source, location=hit.location, quote=quote)

    @staticmethod
    def _citation_supported(citation: Citation, hits: list[RetrievedChunk]) -> bool:
        quote = re.sub(r"\s+", "", citation.quote)
        if not citation.source or not quote:
            return False
        return any(
            citation.source == hit.source and quote in re.sub(r"\s+", "", hit.content)
            for hit in hits
        )

    @staticmethod
    def _fallback_study_advice() -> list[str]:
        return ["结合资料依据复述答案，并用一个例子检查自己是否理解。"]

    @staticmethod
    def _answer_supported_by_context(
        result: AnswerResult, hits: list[RetrievedChunk]
    ) -> bool:
        answer_text = f"{result.direct_answer} {result.explanation}"
        context_text = "\n".join(hit.content for hit in hits)
        stop_chars = set("的是了和与及或在中为对把将有无个一种当前资料相关信息可以使用")
        answer_chars = {
            char
            for char in answer_text
            if "\u4e00" <= char <= "\u9fff" and char not in stop_chars
        }
        context_chars = {
            char
            for char in context_text
            if "\u4e00" <= char <= "\u9fff" and char not in stop_chars
        }
        matched_chars = answer_chars & context_chars

        answer_terms = {
            term.lower()
            for term in re.findall(r"[A-Za-z0-9_]{2,}", answer_text)
        }
        context_terms = {
            term.lower()
            for term in re.findall(r"[A-Za-z0-9_]{2,}", context_text)
        }
        matched_terms = answer_terms & context_terms

        required_chars = min(4, max(2, len(answer_chars) // 3))
        return len(matched_chars) >= required_chars or bool(matched_terms)

    @staticmethod
    def _unknown_answer() -> AnswerResult:
        return AnswerResult(
            found=False,
            direct_answer="当前资料中未找到相关信息。",
            explanation="",
            citations=[],
            study_advice=[],
            confidence="低",
        )

    def ask(self, question: str, session_id: str = "default") -> dict:
        question = self._sanitize_text(question)
        if not question:
            return {
                "result": self._unknown_answer(),
                "markdown": self._unknown_answer().to_markdown(),
                "json": self._unknown_answer().model_dump_json(indent=2),
                "hits": [],
                "standalone_query": "",
            }
        history = self._history(session_id)
        history_text = self._history_text(history)
        standalone = question.strip()
        if history.messages:
            standalone = self.rewrite_chain.invoke(
                {"history": history_text, "question": question}
            ).strip()
        hits = self.rag.search(standalone, top_k=self.settings.top_k)
        if not hits or hits[0].score < self.settings.score_threshold:
            result = self._unknown_answer()
        else:
            raw = self.answer_chain.invoke(
                {
                    "format_instructions": self.answer_parser.get_format_instructions(),
                    "history": history_text,
                    "question": question,
                    "context": self._context(hits),
                }
            )
            result = self._parse(raw, self.answer_parser, AnswerResult)
            if result.found:
                self._ensure_citations(result, hits)
                if not result.citations and self._answer_supported_by_context(result, hits):
                    fallback = self._fallback_citation(hits)
                    if fallback:
                        result.citations = [fallback]
                if result.citations and not result.study_advice:
                    result.study_advice = self._fallback_study_advice()
            # Keep refusing when retrieval is weak, the model says unknown, or the
            # claimed answer has no observable support in the retrieved context.
            if not result.found or not result.citations:
                result = self._unknown_answer()
        markdown = result.to_markdown()
        history.add_messages([HumanMessage(content=question), AIMessage(content=markdown)])
        return {
            "result": result,
            "markdown": markdown,
            "json": result.model_dump_json(indent=2),
            "hits": hits,
            "standalone_query": standalone,
        }

    def clear_history(self, session_id: str) -> None:
        self._history(session_id).clear()

    def generate_outline(
        self, chapter: str, source: str | None = None, **_: Any
    ) -> dict:
        chapter = self._sanitize_text(chapter)
        if not chapter:
            raise ValueError("章节名称不能为空。")
        query = f"{chapter} 核心概念 关键步骤 易错点 复习重点"
        hits = self.rag.search(query, top_k=10, source=source)
        if not hits or hits[0].score < self.settings.score_threshold:
            raise ValueError("当前资料中未找到该章节，无法生成复习提纲。")
        raw = self.outline_chain.invoke(
            {
                "format_instructions": self.outline_parser.get_format_instructions(),
                "scope": source or "全部课程资料",
                "chapter": chapter,
                "context": self._context(hits),
            }
        )
        result = self._parse(raw, self.outline_parser, OutlineResult)
        self._ensure_citations(result, hits)
        has_section = any(
            section.title.strip()
            and any(point.strip() for point in section.key_points)
            for section in result.sections
        )
        has_check_item = any(item.strip() for item in result.review_checklist)
        if not has_section or not has_check_item or not result.citations:
            raise ValueError(
                "模型未生成完整提纲：至少需要一个章节、一个检查项和一个有效资料引用，请重试。"
            )
        return {
            "result": result,
            "markdown": result.to_markdown(),
            "json": result.model_dump_json(indent=2),
            "hits": hits,
        }

    def generate_quiz(
        self,
        chapter: str,
        count: int = 5,
        difficulty: str = "中等",
        question_types: str = "选择题、简答题",
        source: str | None = None,
        **_: Any,
    ) -> dict:
        chapter = self._sanitize_text(chapter)
        if not chapter:
            raise ValueError("章节名称不能为空。")
        count = max(1, min(int(count), 20))
        query = f"{chapter} 定义 原理 步骤 示例 注意事项"
        hits = self.rag.search(query, top_k=12, source=source)
        if not hits or hits[0].score < self.settings.score_threshold:
            raise ValueError("当前资料中未找到该章节，无法生成练习题。")
        raw = self.quiz_chain.invoke(
            {
                "format_instructions": self.quiz_parser.get_format_instructions(),
                "scope": source or "全部课程资料",
                "chapter": chapter,
                "count": count,
                "difficulty": difficulty,
                "question_types": question_types,
                "context": self._context(hits),
            }
        )
        result = self._parse(raw, self.quiz_parser, QuizResult)
        valid_sources = {hit.source for hit in hits}
        fallback = hits[0]
        result.questions = result.questions[:count]
        fallback_citation = self._fallback_citation(hits) or Citation(
            source=fallback.source,
            location=fallback.location,
            quote=fallback.content[:120].replace("\n", " "),
        )
        for index, question in enumerate(result.questions, 1):
            question.number = index
            if (
                question.citation.source not in valid_sources
                or not self._citation_supported(question.citation, hits)
            ):
                question.citation = fallback_citation
        return {
            "result": result,
            "markdown": result.to_markdown(show_answers=True),
            "json": result.model_dump_json(indent=2),
            "hits": hits,
        }

    def invoke(self, mode: str, **params: Any) -> dict:
        return self.workflow.invoke({"mode": mode, "params": params})
