"""PyQt5 desktop interface for the course material assistant."""

from __future__ import annotations

import sys
import uuid
import re
from pathlib import Path
from typing import Any, Callable

from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
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
        except Exception as exc:
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
        self.setWindowTitle("课程资料问答智能体")
        self.resize(1500, 920)
        self.setMinimumSize(1140, 740)
        self._build_ui()
        self._apply_style()
        if demo:
            self._populate_demo()
        else:
            self._set_busy(True, "正在初始化向量索引与知识库…")
            self._run_task(self._create_services, self._services_ready)

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_header())
        root_layout.addWidget(self._make_separator())

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(16, 12, 16, 8)
        body_layout.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._knowledge_panel())
        splitter.addWidget(self._center_panel())
        splitter.addWidget(self._evidence_panel())
        splitter.setSizes([240, 860, 340])
        splitter.setHandleWidth(2)
        body_layout.addWidget(splitter, 1)

        root_layout.addWidget(body, 1)
        root_layout.addWidget(self._make_separator())
        root_layout.addWidget(self._build_status_bar())
        self.setCentralWidget(root)

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("headerBar")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 14, 20, 14)

        title = QLabel("课程资料问答智能体")
        title.setObjectName("headerTitle")
        subtitle = QLabel("基于 RAG 的课程资料检索与智能问答系统")
        subtitle.setObjectName("headerSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addStretch()

        return header

    def _make_separator(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setObjectName("separator")
        return line

    def _knowledge_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("sidePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 16, 14, 14)
        layout.setSpacing(10)

        heading = QLabel("知识库管理")
        heading.setObjectName("panelHeading")
        layout.addWidget(heading)

        self.material_count = QLabel("尚未导入资料")
        self.material_count.setObjectName("captionText")
        layout.addWidget(self.material_count)

        self.material_list = QListWidget()
        self.material_list.setObjectName("materialList")
        self.material_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.material_list.setWordWrap(True)
        self.material_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.material_list.setMinimumHeight(120)
        layout.addWidget(self.material_list, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.import_button = QPushButton("  导入资料")
        self.import_button.setObjectName("actionBtn")
        self.import_button.clicked.connect(self.import_materials)
        self.remove_button = QPushButton("  删除")
        self.remove_button.setObjectName("dangerBtn")
        self.remove_button.clicked.connect(self.remove_material)
        btn_row.addWidget(self.import_button)
        btn_row.addWidget(self.remove_button)
        layout.addLayout(btn_row)

        self.rebuild_button = QPushButton("重建向量索引")
        self.rebuild_button.setObjectName("secondaryBtn")
        self.rebuild_button.clicked.connect(self.rebuild_index)
        layout.addWidget(self.rebuild_button)

        tip = QLabel("支持格式：TXT · Markdown · PDF · DOCX")
        tip.setObjectName("captionText")
        tip.setWordWrap(True)
        layout.addWidget(tip)

        return panel

    def _center_panel(self) -> QWidget:
        self.tabs = QTabWidget()
        self.tabs.setObjectName("mainTabs")
        self.tabs.addTab(self._chat_tab(), "  问答  ")
        self.tabs.addTab(self._outline_tab(), "  提纲  ")
        self.tabs.addTab(self._quiz_tab(), "  练习  ")
        return self.tabs

    def _chat_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(10)

        self.chat_view = QTextBrowser()
        self.chat_view.setObjectName("chatView")
        self.chat_view.setOpenExternalLinks(True)
        self.chat_markdown = (
            "## 欢迎使用课程资料问答助手\n\n"
            "本系统基于 **检索增强生成（RAG）** 技术，从课程资料中检索相关片段并生成有据可查的回答。\n\n"
            "**使用方法：**\n"
            "1. 在左侧导入课程资料（PDF / Markdown / TXT / DOCX）\n"
            "2. 输入问题，系统将检索资料并生成结构化回答\n"
            "3. 右侧面板展示检索到的原始资料片段及其相似度\n\n"
            "您也可以切换到「提纲」或「练习」标签页，生成章节复习提纲与练习题。"
        )
        self.chat_view.setMarkdown(self.chat_markdown)
        layout.addWidget(self.chat_view, 1)

        input_row = QHBoxLayout()
        input_row.setSpacing(8)
        self.question_input = QTextEdit()
        self.question_input.setObjectName("questionInput")
        self.question_input.setPlaceholderText("输入问题，例如：Python 中列表和元组有什么区别？")
        self.question_input.setMaximumHeight(80)
        input_row.addWidget(self.question_input, 1)

        btn_box = QVBoxLayout()
        btn_box.setSpacing(4)
        self.send_button = QPushButton("发送")
        self.send_button.setObjectName("primaryBtn")
        self.send_button.setFixedWidth(72)
        self.send_button.clicked.connect(self.ask_question)
        clear = QPushButton("新建")
        clear.setObjectName("secondaryBtn")
        clear.setFixedWidth(72)
        clear.clicked.connect(self.clear_chat)
        btn_box.addWidget(self.send_button)
        btn_box.addWidget(clear)
        input_row.addLayout(btn_box)

        layout.addLayout(input_row)

        export_row = QHBoxLayout()
        export_row.addStretch()
        export = QPushButton("导出结果")
        export.setObjectName("linkBtn")
        export.clicked.connect(self.export_result)
        export_row.addWidget(export)
        layout.addLayout(export_row)

        return tab

    def _outline_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(10)

        form = QHBoxLayout()
        form.setSpacing(8)
        form.addWidget(self._form_label("资料"))
        self.outline_source = QComboBox()
        self.outline_source.setObjectName("formCombo")
        self.outline_source.addItem("全部资料", None)
        self.outline_source.currentIndexChanged.connect(lambda _: self._refresh_chapter_choices())
        form.addWidget(self.outline_source)
        form.addWidget(self._form_label("章节"))
        self.outline_chapter = QComboBox()
        self.outline_chapter.setObjectName("formCombo")
        self.outline_chapter.setEditable(True)
        self.outline_chapter.setInsertPolicy(QComboBox.NoInsert)
        self.outline_chapter.lineEdit().setPlaceholderText("选择或输入章节名称")
        form.addWidget(self.outline_chapter, 1)
        generate = QPushButton("生成提纲")
        generate.setObjectName("primaryBtn")
        generate.clicked.connect(self.generate_outline)
        form.addWidget(generate)
        layout.addLayout(form)

        self.outline_view = QTextBrowser()
        self.outline_view.setObjectName("chatView")
        self.outline_view.setMarkdown(
            "## 章节复习提纲\n\n"
            "选择资料范围并输入章节名称后，点击「生成提纲」。\n\n"
            "系统将检索相关资料片段，生成包含 **核心概念、关键步骤、易错点** 和 **复习检查清单** 的结构化提纲。"
        )
        layout.addWidget(self.outline_view, 1)

        export_row = QHBoxLayout()
        export_row.addStretch()
        export = QPushButton("导出提纲")
        export.setObjectName("linkBtn")
        export.clicked.connect(self.export_result)
        export_row.addWidget(export)
        layout.addLayout(export_row)

        return tab

    def _quiz_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(10)

        form = QHBoxLayout()
        form.setSpacing(6)
        form.addWidget(self._form_label("资料"))
        self.quiz_source = QComboBox()
        self.quiz_source.setObjectName("formCombo")
        self.quiz_source.addItem("全部资料", None)
        self.quiz_source.currentIndexChanged.connect(lambda _: self._refresh_chapter_choices())
        form.addWidget(self.quiz_source)
        form.addWidget(self._form_label("章节"))
        self.quiz_chapter = QComboBox()
        self.quiz_chapter.setObjectName("formCombo")
        self.quiz_chapter.setEditable(True)
        self.quiz_chapter.setInsertPolicy(QComboBox.NoInsert)
        self.quiz_chapter.lineEdit().setPlaceholderText("选择或输入章节")
        form.addWidget(self.quiz_chapter, 1)
        form.addWidget(self._form_label("题量"))
        self.quiz_count = QSpinBox()
        self.quiz_count.setObjectName("formSpin")
        self.quiz_count.setRange(1, 20)
        self.quiz_count.setValue(2)
        form.addWidget(self.quiz_count)
        form.addWidget(self._form_label("难度"))
        self.quiz_difficulty = QComboBox()
        self.quiz_difficulty.setObjectName("formCombo")
        self.quiz_difficulty.addItems(["基础", "中等", "提高"])
        form.addWidget(self.quiz_difficulty)
        self.quiz_types = QComboBox()
        self.quiz_types.setObjectName("formCombo")
        self.quiz_types.addItems(["选择题、简答题", "选择题", "判断题、简答题", "综合题"])
        form.addWidget(self.quiz_types)
        generate = QPushButton("生成练习")
        generate.setObjectName("primaryBtn")
        generate.clicked.connect(self.generate_quiz)
        form.addWidget(generate)
        layout.addLayout(form)

        self.quiz_view = QTextBrowser()
        self.quiz_view.setObjectName("chatView")
        self.quiz_view.setMarkdown(
            "## 练习题与参考答案\n\n"
            "设置章节、题量和难度后，点击「生成练习」。\n\n"
            "系统将根据课程资料生成包含 **题目、选项、参考答案、解析** 和 **资料依据** 的结构化练习题。"
        )
        layout.addWidget(self.quiz_view, 1)

        export_row = QHBoxLayout()
        export_row.addStretch()
        export = QPushButton("导出练习题")
        export.setObjectName("linkBtn")
        export.clicked.connect(self.export_result)
        export_row.addWidget(export)
        layout.addLayout(export_row)

        return tab

    def _evidence_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("sidePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 16, 14, 14)
        layout.setSpacing(8)

        heading = QLabel("检索证据")
        heading.setObjectName("panelHeading")
        layout.addWidget(heading)

        hint = QLabel("按相似度降序展示原始资料片段，点击条目查看详情")
        hint.setObjectName("captionText")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.evidence_list = QListWidget()
        self.evidence_list.setObjectName("evidenceList")
        self.evidence_list.setWordWrap(True)
        self.evidence_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.evidence_list.currentRowChanged.connect(self.show_evidence)
        layout.addWidget(self.evidence_list, 1)

        self.evidence_text = QTextBrowser()
        self.evidence_text.setObjectName("evidenceText")
        self.evidence_text.setMaximumHeight(260)
        layout.addWidget(self.evidence_text)

        return panel

    def _form_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("formLabel")
        return label

    def _build_status_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("statusBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 6, 20, 6)

        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("statusText")
        layout.addWidget(self.status_label)

        layout.addStretch()

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedWidth(140)
        self.progress.setFixedHeight(4)
        self.progress.hide()
        layout.addWidget(self.progress)

        return bar

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            /* ── Global ── */
            QMainWindow { background: #f2f6fa; }
            QWidget { background: transparent; color: #2d3436; font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif; font-size: 10pt; }

            /* ── Header ── */
            QWidget#headerBar { background: #4a8ec9; }
            QLabel#headerTitle { font-size: 16px; font-weight: 700; color: #ffffff; background: transparent; }
            QLabel#headerSubtitle { font-size: 10pt; color: #d0e2f3; background: transparent; }

            /* ── Separator ── */
            QFrame#separator { background: #c8dbea; max-height: 1px; }

            /* ── Side panels ── */
            QWidget#sidePanel { background: #ffffff; border: 1px solid #d8e2ec; border-radius: 6px; }
            QLabel#panelHeading { font-size: 13px; font-weight: 700; color: #3a7ab5; background: transparent; padding-bottom: 4px; border-bottom: 2px solid #4a8ec9; }
            QLabel#captionText { font-size: 9pt; color: #7a8a9a; background: transparent; }

            /* ── Lists ── */
            QListWidget#materialList, QListWidget#evidenceList {
                background: #f8fafc; border: 1px solid #d8e2ec; border-radius: 4px;
                padding: 4px; font-size: 9.5pt;
            }
            QListWidget#materialList::item, QListWidget#evidenceList::item {
                padding: 6px 8px; border-bottom: 1px solid #edf2f8; border-radius: 3px;
            }
            QListWidget#materialList::item:selected, QListWidget#evidenceList::item:selected {
                background: #dce8f4; color: #2a5a8a;
            }
            QListWidget#materialList::item:hover, QListWidget#evidenceList::item:hover {
                background: #edf2f8;
            }

            /* ── Buttons ── */
            QPushButton {
                background: #edf2f8; border: 1px solid #d0dce8; border-radius: 4px;
                padding: 6px 14px; font-size: 9.5pt; color: #3d4f5f;
            }
            QPushButton:hover { background: #dce8f4; }
            QPushButton:pressed { background: #c8dbea; }
            QPushButton:disabled { background: #edf2f8; color: #a0b0c0; border-color: #d8e2ec; }

            QPushButton#primaryBtn {
                background: #4a8ec9; color: #ffffff; border: none; font-weight: 600;
                padding: 7px 18px; border-radius: 4px;
            }
            QPushButton#primaryBtn:hover { background: #3d7ab5; }
            QPushButton#primaryBtn:pressed { background: #346aa0; }
            QPushButton#primaryBtn:disabled { background: #8ab0d4; color: #d0e2f3; }

            QPushButton#secondaryBtn {
                background: #f2f6fa; border: 1px solid #d0dce8; color: #5a6a7a;
            }
            QPushButton#secondaryBtn:hover { background: #e4ecf4; }

            QPushButton#dangerBtn {
                background: #f2f6fa; border: 1px solid #d0dce8; color: #8b4513;
            }
            QPushButton#dangerBtn:hover { background: #f5ebe0; border-color: #c49a6c; }

            QPushButton#actionBtn {
                background: #4a8ec9; color: #ffffff; border: none; font-weight: 600;
            }
            QPushButton#actionBtn:hover { background: #3d7ab5; }

            QPushButton#linkBtn {
                background: transparent; border: none; color: #4a8ec9;
                font-size: 9pt; padding: 2px 6px;
            }
            QPushButton#linkBtn:hover { color: #346aa0; text-decoration: underline; }

            /* ── Tabs ── */
            QTabWidget#mainTabs::pane {
                background: #ffffff; border: 1px solid #d8e2ec; border-radius: 6px;
                top: -1px;
            }
            QTabBar::tab {
                background: #e8eef5; color: #5a6a7a; border: 1px solid #d8e2ec;
                padding: 8px 24px; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px;
                font-size: 10pt;
            }
            QTabBar::tab:selected {
                background: #ffffff; color: #3a7ab5; font-weight: 600;
                border-bottom: 2px solid #4a8ec9;
            }
            QTabBar::tab:hover:!selected { background: #dce8f4; }

            /* ── Text areas ── */
            QTextBrowser#chatView {
                background: #f8fafc; border: 1px solid #d8e2ec; border-radius: 6px;
                padding: 16px 20px; font-size: 10pt; line-height: 1.6;
                selection-background-color: #4a8ec9; selection-color: #ffffff;
            }
            QTextEdit#questionInput {
                background: #ffffff; border: 2px solid #d0dce8; border-radius: 6px;
                padding: 8px 12px; font-size: 10pt; color: #2d3436;
            }
            QTextEdit#questionInput:focus { border-color: #4a8ec9; }
            QTextBrowser#evidenceText {
                background: #f8fafc; border: 1px solid #d8e2ec; border-radius: 4px;
                padding: 10px 12px; font-size: 9.5pt;
            }

            /* ── Form controls ── */
            QComboBox, QSpinBox {
                background: #ffffff; border: 1px solid #d0dce8; border-radius: 4px;
                padding: 5px 28px 5px 8px; font-size: 9.5pt; color: #2d3436;
            }
            QComboBox:hover, QSpinBox:hover { border-color: #4a8ec9; }
            QComboBox::drop-down {
                subcontrol-origin: padding; subcontrol-position: center right;
                width: 24px; border: none; background: transparent;
            }
            QComboBox::down-arrow {
                image: none; border-left: 4px solid transparent; border-right: 4px solid transparent;
                border-top: 5px solid #7a8a9a; width: 0; height: 0;
            }
            QComboBox::down-arrow:hover { border-top-color: #3a7ab5; }
            QComboBox QAbstractItemView {
                background: #ffffff; border: 1px solid #d0dce8; selection-background-color: #dce8f4;
                selection-color: #2a5a8a; outline: none; padding: 2px;
            }
            QLabel#formLabel { font-size: 9pt; color: #7a8a9a; background: transparent; padding-right: 2px; }

            /* ── Status bar ── */
            QWidget#statusBar { background: #ffffff; border-top: 1px solid #d8e2ec; }
            QLabel#statusText { font-size: 9pt; color: #7a8a9a; background: transparent; }
            QProgressBar { border: none; background: #e8eef5; border-radius: 2px; }
            QProgressBar::chunk { background: #4a8ec9; border-radius: 2px; }

            /* ── Scrollbar ── */
            QScrollBar:vertical { width: 6px; background: transparent; margin: 0; }
            QScrollBar::handle:vertical { background: #c8dbea; border-radius: 3px; min-height: 30px; }
            QScrollBar::handle:vertical:hover { background: #a8c4da; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
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
            self._set_busy(False, "未检测到 API Key，请在 .env 中配置后重启")
        else:
            self._set_busy(False, "就绪")

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
        QMessageBox.critical(self, "错误", message)

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
            chunks = item.get("chunks", 0)
            size_kb = item.get("size", 0) / 1024
            label = f"{item['name']}\n{chunks} 个片段 · {size_kb:.0f} KB"
            self.material_list.addItem(label)
            self.outline_source.addItem(item["name"], item["name"])
            self.quiz_source.addItem(item["name"], item["name"])
        count = len(records)
        self.material_count.setText(f"已导入 {count} 份资料" if count else "尚未导入资料")
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
        added, skipped = len(result["added"]), len(result["skipped"])
        self._set_busy(False, f"导入完成：{added} 份新增，{skipped} 份已跳过")
        if result.get("errors"):
            QMessageBox.warning(self, "导入提示", "\n".join(result["errors"]))

    def remove_material(self) -> None:
        item = self.material_list.currentItem()
        if not item or not self.rag:
            return
        filename = item.text().splitlines()[0]
        self._set_busy(True, "正在删除并重建索引…")
        self._run_task(self.rag.remove_material, lambda _: self._remove_done(), filename)

    def _remove_done(self) -> None:
        self.refresh_materials()
        self._set_busy(False, "资料已删除，索引已更新")

    def rebuild_index(self) -> None:
        if not self.rag:
            return
        self._set_busy(True, "正在重建向量索引…")
        self._run_task(self.rag.rebuild_index, self._rebuild_done)

    def _rebuild_done(self, manifest: dict) -> None:
        self.refresh_materials()
        total = manifest.get("total_chunks", 0)
        self._set_busy(False, f"索引重建完成，共 {total} 个片段")

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
            f"\n\n---\n\n### Q\n\n{shown_question}\n\n### A\n\n"
            f"{self._PENDING_ANSWER}"
        )
        self.chat_view.setMarkdown(self.chat_markdown)
        self.question_input.clear()
        self._set_busy(True, "正在检索资料并生成回答…")
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
        self.chat_markdown = (
            "## 新会话\n\n"
            "对话历史已清空，可以开始新的资料问答。"
        )
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
        self._set_busy(False, "提纲生成完成")

    def generate_quiz(self) -> None:
        chapter = self.quiz_chapter.currentText().strip()
        if not chapter:
            return
        if not self.assistant:
            QMessageBox.information(self, "模型未配置", "请检查项目 .env 后重新启动。")
            return
        self._set_busy(True, "正在生成练习题…")
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
        for i, hit in enumerate(hits, 1):
            item = QListWidgetItem(f"[{i}]  {hit.display_label()}")
            self.evidence_list.addItem(item)
        if hits:
            self.evidence_list.setCurrentRow(0)

    def show_evidence(self, row: int) -> None:
        hits = self.current_hits
        if 0 <= row < len(hits):
            hit = hits[row]
            self.evidence_text.setMarkdown(
                f"**{hit.source}**\n\n"
                f"位置：{hit.location}  \n"
                f"相似度：`{hit.score:.3f}`\n\n"
                f"---\n\n{hit.content}"
            )

    def export_result(self) -> None:
        if not self.last_payload:
            QMessageBox.information(self, "暂无结果", "请先完成一次问答、提纲或习题生成。")
            return
        path, selected = QFileDialog.getSaveFileName(
            self, "导出结构化结果", str(self.settings.output_dir / "文档.md"),
            "Markdown (*.md);;JSON (*.json)"
        )
        if not path:
            return
        key = "json" if selected.startswith("JSON") or path.lower().endswith(".json") else "markdown"
        Path(path).write_text(self.last_payload[key], encoding="utf-8")
        self.status_label.setText(f"已导出至：{path}")

    def _populate_demo(self) -> None:
        self.material_list.addItems(
            [
                "Python程序设计课程讲义.md\n12 个片段 · 48 KB",
                "Python实验指导书.md\n7 个片段 · 23 KB",
                "RAG资料库(1).pdf\n4 个片段 · 156 KB",
            ]
        )
        for name in ("Python程序设计课程讲义.md", "Python实验指导书.md", "RAG资料库(1).pdf"):
            self.outline_source.addItem(name, name)
            self.quiz_source.addItem(name, name)
        for chapter in ("第一章 Python 基础与运行方式", "第二章 条件判断与循环", "第三章 组合数据类型", "第四章 函数"):
            self.outline_chapter.addItem(chapter)
            self.quiz_chapter.addItem(chapter)
        self.material_count.setText("已导入 3 份资料")
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
        self.chat_markdown = "## 示例问答\n\n### Q\n\n列表和元组有什么区别？\n\n" + demo_markdown
        self.chat_view.setMarkdown(self.chat_markdown)
        self.evidence_list.addItem("[1]  Python程序设计课程讲义.md · 第三章 · 0.892")
        self.evidence_text.setMarkdown(
            "**Python程序设计课程讲义.md**\n\n"
            "相似度：`0.892`\n\n"
            "---\n\n"
            "列表属于可变序列，可以修改、添加和删除元素；元组属于不可变序列。"
        )
        self.status_label.setText("演示模式")


def run_app() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()
