from config import Settings


def test_model_generation_defaults():
    settings = Settings()
    assert settings.enable_thinking is False
    assert settings.max_tokens == 3000
    assert settings.embedding_model == "embedding-3"
    assert settings.embedding_base_url == "https://open.bigmodel.cn/api/paas/v4/embeddings"
    assert settings.embedding_client_base_url == "https://open.bigmodel.cn/api/paas/v4"
