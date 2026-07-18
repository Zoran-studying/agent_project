import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from ui import MainWindow


def test_demo_window_can_be_created():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(demo=True)
    assert "课程资料问答智能体" in window.windowTitle()
    assert window.material_list.count() == 3
    assert window.quiz_source.count() == 3
    assert "直接答案" in window.chat_view.toPlainText()
    window.close()
    app.processEvents()


def test_answer_rendering_does_not_show_markdown_markers():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(demo=True)
    window.chat_markdown = (
        "# 会话\n\n---\n\n### Q\n\n你是什么模型？\n\n### A\n\n"
        + window._PENDING_ANSWER
    )

    window._answer_ready({"markdown": "当前资料中未找到相关信息。", "hits": []})

    plain_text = window.chat_view.toPlainText()
    assert "当前资料中未找到相关信息。" in plain_text
    assert "###" not in plain_text
    assert "---" not in plain_text
    window.close()
    app.processEvents()
