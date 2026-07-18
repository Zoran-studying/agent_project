"""PyQt5 desktop interface for the course material assistant."""

from __future__ import annotations

import sys
import uuid
import re
from pathlib import Path
from typing import Any, Callable

from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from agent import CourseAssistant
from config import Settings
from rag import RAGService
from tools import configure_services


class TaskThread(QThread):
    result_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, function: Callable[..., Any], *args: Any, **kwargs: Any):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            self.result_ready.emit(self.function(*self.args, **self.kwargs))
        except Exception as exc:  # UI boundary: show actionable error to the user.
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    _PENDING_ANSWER = "<!-- COURSE_ASSISTANT_PENDING -->"

    def __init__(self, demo: bool = False):
        super().__init__()
        self.demo = demo
        self.settings = Settings()
        self.rag: RAGService | None = None
        self.assistant: CourseAssistant | None = None
        self.session_id = str(uuid.uuid4())
        self._threads: list[TaskThread] = []
        self.current_hits: list = []
        self.last_payload: dict | None = None
        self.setWindowTitle("课程资料问答智能体 · LangChain RAG")
        self.resize(1460, 900)
        self.setMinimumSize(1120, 720)
        self._build_ui()
        self._apply_style()
        if demo:
            self._populate_demo()
        else:
            self._set_busy(True, "正在初始化本地向量模型与知识库…")
            self._run_task(self._create_services, self._services_ready)

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 12, 14, 12)

        header = QHBoxLayout()
        title = QLabel("课程资料问答智能体")
        title.setObjectName("title")
        subtitle = QLabel("LangChain · 多资料 RAG · 有据可查")
        subtitle.setObjectName("subtitle")
        title_box = QVBoxLayout()
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()
        root_layout.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._knowledge_panel())
        splitter.addWidget(self._center_panel())
        splitter.addWidget(self._evidence_panel())
        splitter.setSizes([260, 850, 350])
        root_layout.addWidget(splitter, 1)

        status_row = QHBoxLayout()
        self.status_label = QLabel("准备就绪")
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumWidth(220)
        self.progress.hide()
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        status_row.addWidget(self.progress)
        root_layout.addLayout(status_row)
        self.setCentralWidget(root)

    def _knowledge_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        heading = QLabel("知识库")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        self.material_count = QLabel("0 份资料")
        self.material_count.setObjectName("muted")
        layout.addWidget(self.material_count)
        self.material_list = QListWidget()
        self.material_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.material_list.setWordWrap(True)
        self.material_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.material_list, 1)
        self.import_button = QPushButton("＋ 导入课程资料")
        self.import_button.clicked.connect(self.import_materials)
        self.remove_button = QPushButton("删除选中资料")
        self.remove_button.clicked.connect(self.remove_material)
        self.rebuild_button = QPushButton("重建向量索引")
        self.rebuild_button.clicked.connect(self.rebuild_index)
        layout.addWidget(self.import_button)
        layout.addWidget(self.remove_button)
        layout.addWidget(self.rebuild_button)
        tip = QLabel("支持 TXT / Markdown / PDF / DOCX\n导入后自动切分、向量化并持久保存")
        tip.setWordWrap(True)
        tip.setObjectName("muted")
        layout.addWidget(tip)
        return panel

    def _center_panel(self) -> QWidget:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._chat_tab(), "资料问答")
        self.tabs.addTab(self._outline_tab(), "章节提纲")
        self.tabs.addTab(self._quiz_tab(), "练习题")
        return self.tabs

    def _chat_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.chat_view = QTextBrowser()
        self.chat_view.setOpenExternalLinks(True)
        self.chat_markdown = (
            "# 欢迎使用课程资料问答智能体\n\n"
            "请先从左侧导入课程资料，然后提出问题。回答将展示直接答案、知识点解释、资料来源和学习建议。"
        )
        self.chat_view.setMarkdown(self.chat_markdown)
        layout.addWidget(self.chat_view, 1)
        self.question_input = QTextEdit()
        self.question_input.setPlaceholderText("例如：Python 中列表和元组有什么区别？")
        self.question_input.setMaximumHeight(100)
        layout.addWidget(self.question_input)
        actions = QHBoxLayout()
        self.send_button = QPushButton("发送问题")
        self.send_button.setObjectName("primary")
        self.send_button.clicked.connect(self.ask_question)
        clear = QPushButton("新建会话")
        clear.clicked.connect(self.clear_chat)
        export = QPushButton("导出结果")
        export.clicked.connect(self.export_result)
        actions.addWidget(self.send_button)
        actions.addWidget(clear)
        actions.addStretch()
        actions.addWidget(export)
        layout.addLayout(actions)
        return tab

    def _outline_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        form = QHBoxLayout()
        self.outline_source = QComboBox()
        self.outline_source.addItem("全部资料", None)
        self.outline_source.currentIndexChanged.connect(lambda _: self._refresh_chapter_choices())
        self.outline_chapter = QComboBox()
        self.outline_chapter.setEditable(True)
        self.outline_chapter.setInsertPolicy(QComboBox.NoInsert)
        self.outline_chapter.lineEdit().setPlaceholderText("选择章节或输入主题")
        generate = QPushButton("生成复习提纲")
        generate.setObjectName("primary")
        generate.clicked.connect(self.generate_outline)
        form.addWidget(QLabel("资料范围"))
        form.addWidget(self.outline_source)
        form.addWidget(QLabel("章节"))
        form.addWidget(self.outline_chapter, 1)
        form.addWidget(generate)
        layout.addLayout(form)
        self.outline_view = QTextBrowser()
        self.outline_view.setMarkdown("# 章节复习提纲\n\n选择资料并输入章节后生成。")
        layout.addWidget(self.outline_view, 1)
        export = QPushButton("导出提纲")
        export.clicked.connect(self.export_result)
        layout.addWidget(export, alignment=Qt.AlignRight)
        return tab

    def _quiz_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        form = QHBoxLayout()
        self.quiz_source = QComboBox()
        self.quiz_source.addItem("全部资料", None)
        self.quiz_source.currentIndexChanged.connect(lambda _: self._refresh_chapter_choices())
        self.quiz_chapter = QComboBox()
        self.quiz_chapter.setEditable(True)
        self.quiz_chapter.setInsertPolicy(QComboBox.NoInsert)
        self.quiz_chapter.lineEdit().setPlaceholderText("选择章节或输入主题")
        self.quiz_count = QSpinBox()
        self.quiz_count.setRange(1, 20)
        self.quiz_count.setValue(5)
        self.quiz_difficulty = QComboBox()
        self.quiz_difficulty.addItems(["基础", "中等", "提高"])
        self.quiz_types = QComboBox()
        self.quiz_types.addItems(["选择题、简答题", "选择题", "判断题、简答题", "综合题"])
        generate = QPushButton("生成练习题")
        generate.setObjectName("primary")
        generate.clicked.connect(self.generate_quiz)
        form.addWidget(QLabel("资料范围"))
        form.addWidget(self.quiz_source)
        form.addWidget(QLabel("章节"))
        form.addWidget(self.quiz_chapter, 1)
        form.addWidget(QLabel("题量"))
        form.addWidget(self.quiz_count)
        form.addWidget(QLabel("难度"))
        form.addWidget(self.quiz_difficulty)
        form.addWidget(self.quiz_types)
        form.addWidget(generate)
        layout.addLayout(form)
        self.quiz_view = QTextBrowser()
        self.quiz_view.setMarkdown("# 练习题与参考答案\n\n设置章节、题量和难度后生成。")
        layout.addWidget(self.quiz_view, 1)
        export = QPushButton("导出练习题")
        export.clicked.connect(self.export_result)
        layout.addWidget(export, alignment=Qt.AlignRight)
        return tab

    def _evidence_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        heading = QLabel("检索证据")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        hint = QLabel("按相关度从高到低展示原始资料片段")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.evidence_list = QListWidget()
        self.evidence_list.setWordWrap(True)
        self.evidence_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.evidence_list.currentRowChanged.connect(self.show_evidence)
        layout.addWidget(self.evidence_list, 1)
        self.evidence_text = QTextBrowser()
        self.evidence_text.setMaximumHeight(280)
        layout.addWidget(self.evidence_text)
        return panel

    def _apply_style(self) -> None:
        self.setFont(QFont("Microsoft YaHei UI", 10))
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #f4f7fb; color: #172033; }
            QWidget#panel { background: #ffffff; border: 1px solid #dfe7f2; border-radius: 12px; }
            QLabel#title { font-size: 25px; font-weight: 700; color: #12213a; }
            QLabel#subtitle, QLabel#muted { color: #6b7890; }
            QLabel#sectionTitle { font-size: 17px; font-weight: 700; }
            QPushButton { background: #eef3fa; border: 1px solid #d4deec; border-radius: 7px; padding: 8px 12px; }
            QPushButton:hover { background: #e1eaf6; }
            QPushButton#primary { background: #3867e8; color: white; border: none; font-weight: 600; }
            QPushButton#primary:hover { background: #2f59ca; }
            QLineEdit, QTextEdit, QTextBrowser, QListWidget, QComboBox, QSpinBox {
                background: white; border: 1px solid #d9e2ef; border-radius: 7px; padding: 6px;
            }
            QTabWidget::pane { background: white; border: 1px solid #dfe7f2; border-radius: 10px; }
            QTabBar::tab { padding: 10px 22px; background: #eaf0f8; margin-right: 3px; border-radius: 6px; }
            QTabBar::tab:selected { background: #3867e8; color: white; }
            """
        )

    def _create_services(self) -> tuple[RAGService, CourseAssistant | None]:
        rag = RAGService(self.settings)
        assistant = (
            CourseAssistant(self.settings, rag) if self.settings.api_key else None
        )
        configure_services(rag, assistant)
        return rag, assistant

    def _services_ready(
        self, services: tuple[RAGService, CourseAssistant | None]
    ) -> None:
        self.rag, self.assistant = services
        self.refresh_materials()
        if self.assistant is None:
            self._set_busy(False, "未配置 API Key；请检查项目 .env 后重新启动")
        else:
            self._set_busy(False, "准备就绪")

    def _run_task(self, function: Callable, callback: Callable, *args: Any) -> None:
        thread = TaskThread(function, *args)
        self._threads.append(thread)
        thread.result_ready.connect(callback)
        thread.failed.connect(self._show_error)
        thread.finished.connect(lambda: self._threads.remove(thread) if thread in self._threads else None)
        thread.start()

    def _set_busy(self, busy: bool, text: str) -> None:
        self.status_label.setText(text)
        self.progress.setVisible(busy)
        for button in (self.import_button, self.send_button, self.rebuild_button):
            button.setEnabled(not busy)

    def _show_error(self, message: str) -> None:
        self._set_busy(False, "操作失败")
        QMessageBox.critical(self, "操作失败", message)

    def refresh_materials(self) -> None:
        if not self.rag:
            return
        records = self.rag.list_materials()
        self.material_list.clear()
        self.outline_source.clear()
        self.outline_source.addItem("全部资料", None)
        self.quiz_source.clear()
        self.quiz_source.addItem("全部资料", None)
        for item in records:
            label = f"{item['name']}\n{item.get('chunks', 0)} 个片段"
            self.material_list.addItem(label)
            self.outline_source.addItem(item["name"], item["name"])
            self.quiz_source.addItem(item["name"], item["name"])
        self.material_count.setText(f"{len(records)} 份资料")
        self._refresh_chapter_choices()

    def _refresh_chapter_choices(self) -> None:
        if not self.rag or not hasattr(self, "outline_chapter") or not hasattr(self, "quiz_chapter"):
            return
        self._set_chapter_choices(
            self.outline_chapter,
            self.rag.list_chapters(self.outline_source.currentData()),
        )
        self._set_chapter_choices(
            self.quiz_chapter,
            self.rag.list_chapters(self.quiz_source.currentData()),
        )

    @staticmethod
    def _set_chapter_choices(combo: QComboBox, chapters: list[str]) -> None:
        current = combo.currentText().strip()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(chapters)
        if current:
            index = combo.findText(current)
            if index >= 0:
                combo.setCurrentIndex(index)
            else:
                combo.setEditText(current)
        elif chapters:
            combo.setCurrentIndex(0)
        else:
            combo.setEditText("")
        combo.blockSignals(False)

    def import_materials(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "导入课程资料", "", "课程资料 (*.txt *.md *.pdf *.docx)"
        )
        if not paths or not self.rag:
            return
        self._set_busy(True, "正在解析并建立向量索引…")
        self._run_task(self.rag.add_materials, self._import_done, paths)

    def _import_done(self, result: dict) -> None:
        self.refresh_materials()
        self._set_busy(False, f"已导入 {len(result['added'])} 份，跳过 {len(result['skipped'])} 份")
        if result.get("errors"):
            QMessageBox.warning(self, "部分文件失败", "\n".join(result["errors"]))

    def remove_material(self) -> None:
        item = self.material_list.currentItem()
        if not item or not self.rag:
            return
        filename = item.text().splitlines()[0]
        self._set_busy(True, "正在删除并重建索引…")
        self._run_task(self.rag.remove_material, lambda _: self._remove_done(), filename)

    def _remove_done(self) -> None:
        self.refresh_materials()
        self._set_busy(False, "资料已删除")

    def rebuild_index(self) -> None:
        if not self.rag:
            return
        self._set_busy(True, "正在重建向量索引…")
        self._run_task(self.rag.rebuild_index, self._rebuild_done)

    def _rebuild_done(self, manifest: dict) -> None:
        self.refresh_materials()
        self._set_busy(False, f"索引已重建，共 {manifest.get('total_chunks', 0)} 个片段")

    def ask_question(self) -> None:
        question = self.question_input.toPlainText().strip()
        if not question:
            return
        if not self.assistant:
            QMessageBox.information(
                self, "模型未配置", "请在项目 .env 中填写 API Key，然后重新启动。"
            )
            return
        shown_question = re.sub(r"([\\`*{}\[\]()#+\-.!_>|])", r"\\\1", question)
        self.chat_markdown += (
            f"\n\n---\n\n### 你\n\n{shown_question}\n\n### 智能体\n\n"
            f"{self._PENDING_ANSWER}"
        )
        self.chat_view.setMarkdown(self.chat_markdown)
        self.question_input.clear()
        self._set_busy(True, "正在检索资料并生成有依据的回答…")
        self._run_task(self.assistant.ask, self._answer_ready, question, self.session_id)

    def _answer_ready(self, payload: dict) -> None:
        self.last_payload = payload
        self.chat_markdown = self.chat_markdown.replace(
            self._PENDING_ANSWER, payload["markdown"], 1
        )
        self.chat_view.setMarkdown(self.chat_markdown)
        self._show_hits(payload.get("hits", []))
        self._set_busy(False, "回答完成")

    def clear_chat(self) -> None:
        if self.assistant:
            self.assistant.clear_history(self.session_id)
        self.session_id = str(uuid.uuid4())
        self.chat_markdown = "# 新会话\n\n已清空对话历史，可以开始新的资料问答。"
        self.chat_view.setMarkdown(self.chat_markdown)
        self.evidence_list.clear()
        self.evidence_text.clear()

    def generate_outline(self) -> None:
        chapter = self.outline_chapter.currentText().strip()
        if not chapter:
            return
        if not self.assistant:
            QMessageBox.information(self, "模型未配置", "请检查项目 .env 后重新启动。")
            return
        self._set_busy(True, "正在生成章节复习提纲…")
        self._run_task(
            self.assistant.generate_outline,
            self._outline_ready,
            chapter,
            self.outline_source.currentData(),
        )

    def _outline_ready(self, payload: dict) -> None:
        self.last_payload = payload
        self.outline_view.setMarkdown(payload["markdown"])
        self._show_hits(payload.get("hits", []))
        self._set_busy(False, "复习提纲生成完成")

    def generate_quiz(self) -> None:
        chapter = self.quiz_chapter.currentText().strip()
        if not chapter:
            return
        if not self.assistant:
            QMessageBox.information(self, "模型未配置", "请检查项目 .env 后重新启动。")
            return
        self._set_busy(True, "正在生成练习题与参考答案…")
        self._run_task(
            self.assistant.generate_quiz,
            self._quiz_ready,
            chapter,
            self.quiz_count.value(),
            self.quiz_difficulty.currentText(),
            self.quiz_types.currentText(),
            self.quiz_source.currentData(),
        )

    def _quiz_ready(self, payload: dict) -> None:
        self.last_payload = payload
        self.quiz_view.setMarkdown(payload["markdown"])
        self._show_hits(payload.get("hits", []))
        self._set_busy(False, "练习题生成完成")

    def _show_hits(self, hits: list) -> None:
        self.evidence_list.clear()
        self.current_hits = hits
        for hit in hits:
            self.evidence_list.addItem(hit.display_label())
        if hits:
            self.evidence_list.setCurrentRow(0)

    def show_evidence(self, row: int) -> None:
        hits = self.current_hits
        if 0 <= row < len(hits):
            hit = hits[row]
            self.evidence_text.setMarkdown(
                f"### {hit.source}\n\n**位置：** {hit.location}  \n"
                f"**相似度：** {hit.score:.3f}\n\n---\n\n{hit.content}"
            )

    def export_result(self) -> None:
        if not self.last_payload:
            QMessageBox.information(self, "暂无结果", "请先完成一次问答、提纲或习题生成。")
            return
        path, selected = QFileDialog.getSaveFileName(
            self, "导出结构化结果", str(self.settings.output_dir / "课程助手结果.md"),
            "Markdown (*.md);;JSON (*.json)"
        )
        if not path:
            return
        key = "json" if selected.startswith("JSON") or path.lower().endswith(".json") else "markdown"
        Path(path).write_text(self.last_payload[key], encoding="utf-8")
        self.status_label.setText(f"已导出：{path}")

    def _populate_demo(self) -> None:
        self.material_list.addItems(
            [
                "Python程序设计课程讲义.md\n12 个片段",
                "Python实验指导书.md\n7 个片段",
                "RAG资料库(1).pdf\n4 个片段",
            ]
        )
        for name in ("Python程序设计课程讲义.md", "Python实验指导书.md", "RAG资料库(1).pdf"):
            self.outline_source.addItem(name, name)
            self.quiz_source.addItem(name, name)
        for chapter in ("第一章 Python 基础与运行方式", "第二章 条件判断与循环", "第三章 组合数据类型", "第四章 函数"):
            self.outline_chapter.addItem(chapter)
            self.quiz_chapter.addItem(chapter)
        self.material_count.setText("3 份资料")
        demo_markdown = """## 直接答案

Python 列表是可变序列，元组是不可变序列。列表适合需要增删改的集合；元组适合结构固定、希望避免误修改的数据。

## 知识点解释

列表使用方括号创建，支持 `append`、`remove` 等修改操作；元组使用圆括号创建，创建后不能修改其中元素。

## 资料依据

- **Python程序设计课程讲义.md · 第三章 组合数据类型**：列表属于可变序列，元组属于不可变序列。

## 学习建议

- 编写一段代码分别尝试修改列表和元组，观察异常信息。

**置信度：高**
"""
        self.chat_markdown = "# 示例问答\n\n### 你\n\n列表和元组有什么区别？\n\n" + demo_markdown
        self.chat_view.setMarkdown(self.chat_markdown)
        self.evidence_list.addItem("Python程序设计课程讲义.md · 第三章 · 相似度 0.892")
        self.evidence_text.setMarkdown(
            "### Python程序设计课程讲义.md\n\n**相似度：** 0.892\n\n"
            "列表属于可变序列，可以修改、添加和删除元素；元组属于不可变序列。"
        )
        self.status_label.setText("演示模式 · 界面验收")


def run_app() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()
