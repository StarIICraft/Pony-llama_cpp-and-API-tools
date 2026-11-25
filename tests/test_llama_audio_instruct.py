import importlib.util
import sys
import types
from pathlib import Path

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))

PACKAGE_NAME = "pony_llama_audio_instruct_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(PLUGIN_ROOT)]
sys.modules[PACKAGE_NAME] = package

spec = importlib.util.spec_from_file_location(
    f"{PACKAGE_NAME}.nodes",
    PLUGIN_ROOT / "nodes.py",
)
nodes = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = nodes
spec.loader.exec_module(nodes)


class FakeLlama:
    def __init__(self):
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "choices": [{
                "message": {
                    "content": (
                        "<|channel>thought\ninternal<channel|>"
                        "<|channel>final\n最终文本"
                    ),
                },
            }],
        }


def config_for(handler):
    return {
        "model": "test.gguf",
        "mmproj": "test-mmproj.gguf",
        "chat_handler": handler,
        "n_ctx": 4096,
        "vram_limit": -1,
        "image_min_tokens": 0,
        "image_max_tokens": 0,
        "load_mtp": False,
        "kv_cache_mode": nodes.KV_CACHE_F16,
        "cpu_moe": False,
        "n_cpu_moe": 0,
    }


def audio_input():
    return {
        "waveform": torch.zeros((1, 1, 160), dtype=torch.float32),
        "sample_rate": 16000,
    }


def run_audio(
    handler,
    image_bundle=None,
    inference_mode="text",
):
    fake = FakeLlama()
    config = config_for(handler)
    original = (
        nodes.LLAMA_CPP_STORAGE.llm,
        nodes.LLAMA_CPP_STORAGE.chat_handler,
        nodes.LLAMA_CPP_STORAGE.current_config,
        nodes.LLAMA_CPP_STORAGE.current_embedding_mode,
    )
    nodes.LLAMA_CPP_STORAGE.llm = fake
    nodes.LLAMA_CPP_STORAGE.chat_handler = object()
    nodes.LLAMA_CPP_STORAGE.current_config = config
    nodes.LLAMA_CPP_STORAGE.current_embedding_mode = False
    try:
        result = nodes.llama_cpp_instruct_adv()._process_impl(
            llama_model=config,
            custom_prompt="转写这段音频",
            system_prompt="",
            inference_mode=inference_mode,
            max_frames=24,
            seed=42,
            parameters={},
            image_bundle=image_bundle,
            queue_handler=None,
            skill="",
            audio=audio_input(),
        )
        return result, fake
    finally:
        (
            nodes.LLAMA_CPP_STORAGE.llm,
            nodes.LLAMA_CPP_STORAGE.chat_handler,
            nodes.LLAMA_CPP_STORAGE.current_config,
            nodes.LLAMA_CPP_STORAGE.current_embedding_mode,
        ) = original


def test_generic_instruct_exposes_audio_input():
    optional = nodes.llama_cpp_instruct_adv.INPUT_TYPES()["optional"]
    assert optional["audio"][0] == "AUDIO"


def test_gemma4_audio_and_text_share_one_message():
    result, fake = run_audio("Gemma4")
    assert result == ("最终文本", ["最终文本"])

    content = fake.calls[0]["messages"][-1]["content"]
    assert [item["type"] for item in content] == [
        "text",
        "input_audio",
    ]
    assert content[0]["text"] == "转写这段音频"
    assert content[1]["input_audio"]["format"] == "wav"


def test_qwen3_asr_uses_audio_system_prompt():
    result, fake = run_audio("Qwen3-ASR")
    assert result[0] == "最终文本"
    system = fake.calls[0]["messages"][0]["content"]
    assert "转写音频" in system
    user_content = fake.calls[0]["messages"][-1]["content"]
    assert any(
        item.get("type") == "input_audio"
        for item in user_content
    )


def test_gemma4_accepts_image_audio_and_text_together():
    result, fake = run_audio(
        "Gemma4",
        image_bundle=[
            torch.ones((8, 8, 3), dtype=torch.float32)
        ],
        inference_mode="images",
    )
    assert result[0] == "最终文本"

    content = fake.calls[0]["messages"][-1]["content"]
    assert [item["type"] for item in content] == [
        "text",
        "input_audio",
        "image_url",
    ]
    assert content[-1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )


def test_non_audio_handler_rejects_audio_input():
    config = config_for("Qwen3.5")
    original = (
        nodes.LLAMA_CPP_STORAGE.llm,
        nodes.LLAMA_CPP_STORAGE.chat_handler,
        nodes.LLAMA_CPP_STORAGE.current_config,
    )
    nodes.LLAMA_CPP_STORAGE.llm = object()
    nodes.LLAMA_CPP_STORAGE.chat_handler = object()
    nodes.LLAMA_CPP_STORAGE.current_config = config
    try:
        try:
            nodes.llama_cpp_instruct_adv()._process_impl(
                llama_model=config,
                custom_prompt="test",
                system_prompt="",
                inference_mode="text",
                max_frames=24,
                seed=0,
                parameters={},
                audio=audio_input(),
            )
        except ValueError as error:
            assert "不支持音频" in str(error)
        else:
            raise AssertionError("expected unsupported audio handler")
    finally:
        (
            nodes.LLAMA_CPP_STORAGE.llm,
            nodes.LLAMA_CPP_STORAGE.chat_handler,
            nodes.LLAMA_CPP_STORAGE.current_config,
        ) = original


def test_standalone_multimodal_node_was_removed():
    assert "PonyGemma4MultimodalInstruct" not in nodes.NODE_CLASS_MAPPINGS
    assert "PonyGemma4MultimodalInstruct" not in (
        nodes.NODE_DISPLAY_NAME_MAPPINGS
    )


if __name__ == "__main__":
    test_generic_instruct_exposes_audio_input()
    test_gemma4_audio_and_text_share_one_message()
    test_qwen3_asr_uses_audio_system_prompt()
    test_gemma4_accepts_image_audio_and_text_together()
    test_non_audio_handler_rejects_audio_input()
    test_standalone_multimodal_node_was_removed()
    print("generic llama audio instruct: OK")
