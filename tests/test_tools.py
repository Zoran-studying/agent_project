import json
from pathlib import Path

from config import Settings
from rag import RAGService
from tools import configure_services, load_course_materials, save_result, search_course_content


def test_registered_tools_can_load_search_and_save(tmp_path, hash_embeddings):
    settings = Settings(
        api_key="test",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        score_threshold=0.0,
    )
    rag = RAGService(settings, embeddings=hash_embeddings)
    configure_services(rag)
    material = tmp_path / "实验.txt"
    material.write_text("文件操作推荐使用 with open 自动关闭文件。", encoding="utf-8")
    loaded = json.loads(load_course_materials.invoke({"paths": [str(material)]}))
    assert loaded["added"] == ["实验.txt"]
    hits = json.loads(search_course_content.invoke({"query": "如何自动关闭文件", "top_k": 2}))
    assert hits[0]["source"] == "实验.txt"
    message = save_result.invoke({"filename": "结果.md", "content": "# 结果"})
    assert "文件已保存" in message
    assert (settings.output_dir / "结果.md").exists()

