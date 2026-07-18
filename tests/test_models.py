import pytest
from pydantic import ValidationError

from models import AnswerResult, Citation, OutlineResult, OutlineSection, QuizQuestion, QuizResult


def test_answer_markdown_contains_required_sections():
    answer = AnswerResult(
        direct_answer="列表可修改，元组不可修改。",
        explanation="二者都是有序序列。",
        citations=[Citation(source="讲义.md", location="第三章", quote="列表是可变序列")],
        study_advice=["分别尝试修改两种序列"],
        confidence="高",
    )
    rendered = answer.to_markdown()
    assert "直接答案" in rendered
    assert "资料依据" in rendered
    assert "讲义.md" in rendered


def test_answer_markdown_hides_empty_explanation_and_keeps_advice():
    answer = AnswerResult(
        direct_answer="列表是可变序列。",
        explanation="",
        citations=[Citation(source="讲义.md", location="第三章", quote="列表是可变序列")],
        study_advice=[],
        confidence="高",
    )
    rendered = answer.to_markdown()
    assert "知识点解释" not in rendered
    assert "学习建议" in rendered
    assert "结合资料依据复述答案" in rendered


def test_outline_and_quiz_render_to_markdown():
    citation = Citation(source="讲义.md", location="第一章", quote="Python 是解释型语言")
    outline = OutlineResult(
        title="第一章提纲",
        overview="基础知识",
        sections=[OutlineSection(title="运行方式", key_points=["解释执行"])],
        review_checklist=["能运行脚本"],
        citations=[citation],
    )
    quiz = QuizResult(
        title="第一章练习",
        difficulty="基础",
        questions=[
            QuizQuestion(
                number=1,
                question_type="简答题",
                question="Python 如何执行？",
                answer="由解释器执行。",
                explanation="程序从上到下执行。",
                citation=citation,
            )
        ],
    )
    assert "复习检查清单" in outline.to_markdown()
    assert "参考答案" in quiz.to_markdown()


def test_outline_requires_section_check_item_and_citation():
    with pytest.raises(ValidationError):
        OutlineResult(
            title="空提纲",
            overview="没有生成有效内容",
            sections=[],
            review_checklist=[],
            citations=[],
        )
