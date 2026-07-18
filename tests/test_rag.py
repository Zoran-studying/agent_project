import json
from pathlib import Path

from config import Settings
from rag import RAGService


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        api_key="test",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        score_threshold=0.0,
    )


def test_stale_manifest_does_not_seed_an_empty_library(tmp_path, hash_embeddings):
    settings = make_settings(tmp_path)
    settings.ensure_directories()
    (settings.data_dir / "index_manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "name": "示例资料.md",
                        "sha256": "old-index",
                        "chunks": 3,
                        "size": 100,
                    }
                ],
                "total_chunks": 3,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rag = RAGService(settings, embeddings=hash_embeddings)

    assert rag.list_materials() == []
    assert rag.search("示例内容") == []


def test_import_search_duplicate_and_remove(tmp_path, hash_embeddings):
    source = tmp_path / "课程讲义.md"
    source.write_text(
        "# Python课程\n\n## 列表\n列表是可变序列，可以添加和删除元素。\n\n"
        "## 元组\n元组是不可变序列。",
        encoding="utf-8",
    )
    rag = RAGService(make_settings(tmp_path), embeddings=hash_embeddings)
    assert rag.list_materials() == []

    result = rag.add_materials([source])
    assert result["added"] == ["课程讲义.md"]
    assert rag.list_materials()[0]["chunks"] >= 1
    duplicate = rag.add_materials([source])
    assert duplicate["skipped"] == ["课程讲义.md"]
    hits = rag.search("列表是否可以修改", top_k=3)
    assert hits
    assert hits[0].source == "课程讲义.md"
    assert 0 <= hits[0].score <= 1
    assert rag.remove_material("课程讲义.md") is True
    assert rag.list_materials() == []


def test_txt_encoding_and_multiple_materials(tmp_path, hash_embeddings):
    one = tmp_path / "a.txt"
    two = tmp_path / "b.md"
    one.write_text("第一章\n条件判断使用 if elif else。", encoding="utf-8")
    two.write_text("# 循环\nfor 适合遍历序列，while 依据条件执行。", encoding="utf-8")
    rag = RAGService(make_settings(tmp_path), embeddings=hash_embeddings)
    result = rag.add_materials([one, two])
    assert len(result["added"]) == 2
    assert len(rag.list_materials()) == 2
    assert rag.search("for循环", top_k=2)


def test_list_chapters_from_index_metadata(tmp_path, hash_embeddings):
    lecture = tmp_path / "课程讲义.md"
    lab = tmp_path / "实验指导.md"
    lecture.write_text(
        "# Python课程\n\n## 第一章 基础\n变量和输入输出。\n\n## 第二章 循环\nfor 和 while。",
        encoding="utf-8",
    )
    lab.write_text(
        "# 实验指导\n\n## 实验一 成绩判断\n使用 if 处理分支。",
        encoding="utf-8",
    )
    rag = RAGService(make_settings(tmp_path), embeddings=hash_embeddings)
    rag.add_materials([lecture, lab])

    assert rag.list_chapters("课程讲义.md") == ["第一章 基础", "第二章 循环"]
    assert "实验一 成绩判断" in rag.list_chapters()
