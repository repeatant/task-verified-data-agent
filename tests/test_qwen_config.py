"""Configuration validation uses fake values without reading credentials."""

import pytest

from dataverify.integrations.qwen_config import QwenClientError, QwenConfig

SECRET = "sk-test-never-a-real-credential"
ENDPOINT = "https://example.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"


def config():
    return QwenConfig(SECRET, ENDPOINT, "qwen-test")


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        "https://coding.dashscope.aliyuncs.com/v1",
        "https://fake.example/compatible-mode/v1",
        "https://dashscope.aliyuncs.com/compatible-mode/v1?key=hidden",
        "https://user:pass@dashscope.aliyuncs.com/compatible-mode/v1",
        "https://dashscope.aliyuncs.com:sk-test-never-a-real-credential/compatible-mode/v1",
    ],
)
def test_reject_nonordinary_or_credential_bearing_endpoint(endpoint):
    with pytest.raises(QwenClientError, match="ORDINARY_QWEN_HTTPS_ENDPOINT_REQUIRED"):
        QwenConfig(SECRET, endpoint)


def test_config_hides_key_and_rejects_subscription_key():
    assert SECRET not in repr(config())
    with pytest.raises(QwenClientError, match="SUBSCRIPTION_KEY_REJECTED"):
        QwenConfig("sk-sp-fake", ENDPOINT)
    with pytest.raises(QwenClientError, match="NOT_CONFIGURED"):
        QwenConfig("", ENDPOINT)
