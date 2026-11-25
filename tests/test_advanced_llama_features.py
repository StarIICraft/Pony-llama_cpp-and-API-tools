import importlib.util
import json
import struct
import sys
import tempfile
import types
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))

PACKAGE_NAME = "pony_advanced_llama_features_test"
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


def test_advanced_loader_defaults_preserve_llama_behavior():
    required = nodes.llama_cpp_model_loader.INPUT_TYPES()["required"]
    assert "Gemma4" in required["chat_handler"][0]
    assert "Gemma4-Thinking" in required["chat_handler"][0]
    assert required["kv_cache_mode"][1]["default"] == nodes.KV_CACHE_F16
    assert required["kv_cache_mode"][0] == nodes.KV_CACHE_MODES
    assert required["cpu_moe"][1]["default"] is False
    assert required["n_cpu_moe"][1]["default"] == 0

    kwargs = {}
    nodes._apply_advanced_llama_options(
        kwargs,
        "Qwen3.6",
        nodes.KV_CACHE_F16,
    )
    assert kwargs == {}


def test_q8_kv_cache_is_passed_as_llama_type_id():
    kwargs = {}
    nodes._apply_advanced_llama_options(
        kwargs,
        "Qwen3.6",
        nodes.KV_CACHE_Q8_Q8,
    )
    assert kwargs == {
        "type_k": 8,
        "type_v": 8,
    }


def test_q4_kv_cache_is_passed_as_llama_type_id():
    kwargs = {}
    nodes._apply_advanced_llama_options(
        kwargs,
        "Qwen3.5",
        nodes.KV_CACHE_Q4_Q4,
    )
    assert kwargs == {
        "type_k": 2,
        "type_v": 2,
    }


def test_mixed_q8_q4_kv_cache_mode():
    kwargs = {}
    nodes._apply_advanced_llama_options(
        kwargs,
        "Qwen3.5",
        nodes.KV_CACHE_Q8_Q4,
    )
    assert kwargs == {
        "type_k": 8,
        "type_v": 2,
    }


def test_qwen36_moe_cpu_offload_options():
    kwargs = {}
    nodes._apply_advanced_llama_options(
        kwargs,
        "Qwen3.6-Thinking",
        nodes.KV_CACHE_F16,
        cpu_moe=True,
        n_cpu_moe=12,
    )
    assert kwargs == {"cpu_moe": True}

    kwargs = {}
    nodes._apply_advanced_llama_options(
        kwargs,
        "Qwen3.6",
        nodes.KV_CACHE_F16,
        n_cpu_moe=12,
    )
    assert kwargs == {"n_cpu_moe": 12}

    try:
        nodes._apply_advanced_llama_options(
            {},
            "Gemma4",
            nodes.KV_CACHE_F16,
            cpu_moe=True,
        )
    except ValueError as error:
        assert "Qwen3.6" in str(error)
    else:
        raise AssertionError("expected non-Qwen3.6 MoE offload to fail")


def test_gemma4_reasoning_budget_is_scoped_to_gemma4():
    parameters = nodes._apply_gemma4_reasoning_budget(
        {"temperature": 0.7, "gemma4_reasoning_budget": 64},
        "Gemma4",
    )
    assert parameters["temperature"] == 0.7
    assert parameters["reasoning_budget"] == 64
    assert parameters["reasoning_start"] == "<|channel>"
    assert parameters["reasoning_end"] == "<channel|>"
    assert "gemma4_reasoning_budget" not in parameters

    thinking_parameters = nodes._apply_gemma4_reasoning_budget(
        {"gemma4_reasoning_budget": 32},
        "Gemma4-Thinking",
    )
    assert thinking_parameters["reasoning_budget"] == 32

    assert nodes._apply_gemma4_reasoning_budget(
        {"gemma4_reasoning_budget": -1},
        "Qwen3.6",
    ) == {}

    try:
        nodes._apply_gemma4_reasoning_budget(
            {"gemma4_reasoning_budget": 64},
            "Qwen3.6",
        )
    except ValueError as error:
        assert "Gemma4" in str(error)
    else:
        raise AssertionError("expected Gemma4-only budget validation")


def test_gemma4_channel_thought_is_removed():
    value = (
        "<|channel>thought\ninternal reasoning<channel|>"
        "<|channel>final\nVisible answer"
    )
    assert nodes.llama_cpp_instruct_adv._remove_thinking(value) == (
        "Visible answer"
    )


def test_text_mode_and_removed_preset_prompt():
    required = nodes.llama_cpp_instruct_adv.INPUT_TYPES()["required"]
    assert required["inference_mode"][0][-1] == "text"
    assert "first image" not in required["inference_mode"][0]
    assert "preset_prompt" not in required


def test_empty_system_prompt_uses_embedded_general_assistant():
    resolved = nodes.llama_cpp_instruct_adv._select_system_prompt(
        "",
        "",
    )
    assert resolved == nodes.DEFAULT_ASSISTANT_SYSTEM_PROMPT
    assert "通用型AI助手" in resolved
    assert "使用用户提示词所用的语言回答" in resolved

    explicit = nodes.llama_cpp_instruct_adv._select_system_prompt(
        "Use this system prompt.",
        "",
    )
    assert explicit == "Use this system prompt."

    skill = nodes.llama_cpp_instruct_adv._select_system_prompt(
        "ignored",
        '{"system_prompt":"Skill prompt"}',
    )
    assert skill == "Skill prompt"


def test_new_llama_fields_have_english_and_chinese_labels():
    expected = {
        "kv_cache_mode",
        "cpu_moe",
        "n_cpu_moe",
    }
    for locale in ("en", "zh", "zh-CN"):
        data = json.loads(
            (
                PLUGIN_ROOT
                / "locales"
                / locale
                / "nodeDefs.json"
            ).read_text(encoding="utf-8")
        )
        assert expected.issubset(
            data["PonyLlamaCppModelLoader"]["inputs"]
        )
        assert "gemma4_reasoning_budget" in (
            data["PonyLlamaCppParameters"]["inputs"]
        )


def test_reads_embedded_gguf_chat_template():
    architecture_key = b"general.architecture"
    architecture_value = b"gemma4"
    size_key = b"general.size_label"
    size_value = b"12B"
    key = b"tokenizer.chat_template"
    value = b"{{ messages }}"
    payload = bytearray(b"GGUF")
    payload += struct.pack("<IQQ", 3, 0, 3)
    for metadata_key, metadata_value in (
        (architecture_key, architecture_value),
        (size_key, size_value),
        (key, value),
    ):
        payload += struct.pack("<Q", len(metadata_key))
        payload += metadata_key
        payload += struct.pack("<I", 8)
        payload += struct.pack("<Q", len(metadata_value))
        payload += metadata_value

    temp_path = None
    with tempfile.NamedTemporaryFile(
        suffix=".gguf",
        delete=False,
    ) as file:
        file.write(payload)
        temp_path = Path(file.name)
    try:
        info = nodes.get_gguf_model_info(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)

    assert info["architecture"] == architecture_value.decode("utf-8")
    assert info["size_label"] == size_value.decode("utf-8")
    assert info["chat_template"] == value.decode("utf-8")


def test_gemma4_handler_rejects_non_gemma_model_architecture():
    nodes._validate_chat_handler_architecture("Gemma4", "gemma4")
    nodes._validate_chat_handler_architecture(
        "Gemma4-Thinking",
        "gemma4",
    )

    try:
        nodes._validate_chat_handler_architecture(
            "Gemma4-Thinking",
            "qwen3vl",
        )
    except ValueError as error:
        assert "qwen3vl" in str(error)
        assert "Gemma4" in str(error)
    else:
        raise AssertionError("expected a model/handler mismatch")


def test_gemma4_gguf_template_registers_media_tags():
    handler = types.SimpleNamespace(
        chat_format="<|image|><|audio|><|video|>",
        _chat_format_parser_tags=[],
    )
    nodes._configure_gemma4_media_tags(
        handler,
        "Gemma4",
        True,
    )
    assert handler._chat_format_parser_tags == [
        "<|image|>",
        "<|audio|>",
        "<|video|>",
    ]

    nodes._configure_gemma4_media_tags(
        handler,
        "Gemma4-Thinking",
        False,
    )
    assert handler._chat_format_parser_tags == []


def test_video_mode_requires_image_frames():
    config = {
        "model": "plain.gguf",
        "mmproj": "None",
        "chat_handler": "None",
        "n_ctx": 2048,
        "vram_limit": -1,
        "image_min_tokens": 0,
        "image_max_tokens": 0,
        "load_mtp": False,
        "kv_cache_mode": nodes.KV_CACHE_F16,
        "cpu_moe": False,
        "n_cpu_moe": 0,
    }
    original = (
        nodes.LLAMA_CPP_STORAGE.llm,
        nodes.LLAMA_CPP_STORAGE.chat_handler,
        nodes.LLAMA_CPP_STORAGE.current_config,
        nodes.LLAMA_CPP_STORAGE.current_embedding_mode,
    )
    nodes.LLAMA_CPP_STORAGE.llm = object()
    nodes.LLAMA_CPP_STORAGE.chat_handler = None
    nodes.LLAMA_CPP_STORAGE.current_config = config
    nodes.LLAMA_CPP_STORAGE.current_embedding_mode = False
    try:
        try:
            nodes.llama_cpp_instruct_adv()._process_impl(
                llama_model=config,
                custom_prompt="分析视频",
                system_prompt="",
                inference_mode="video",
                max_frames=24,
                seed=0,
                parameters={},
                image_bundle=None,
                queue_handler=None,
                skill="",
            )
        except ValueError as error:
            assert "IMAGE" in str(error)
            assert "队列处理器" in str(error)
        else:
            raise AssertionError("video mode should require IMAGE frames")
    finally:
        (
            nodes.LLAMA_CPP_STORAGE.llm,
            nodes.LLAMA_CPP_STORAGE.chat_handler,
            nodes.LLAMA_CPP_STORAGE.current_config,
            nodes.LLAMA_CPP_STORAGE.current_embedding_mode,
        ) = original


if __name__ == "__main__":
    test_advanced_loader_defaults_preserve_llama_behavior()
    test_q8_kv_cache_is_passed_as_llama_type_id()
    test_q4_kv_cache_is_passed_as_llama_type_id()
    test_mixed_q8_q4_kv_cache_mode()
    test_qwen36_moe_cpu_offload_options()
    test_gemma4_reasoning_budget_is_scoped_to_gemma4()
    test_gemma4_channel_thought_is_removed()
    test_text_mode_and_removed_preset_prompt()
    test_empty_system_prompt_uses_embedded_general_assistant()
    test_new_llama_fields_have_english_and_chinese_labels()
    test_reads_embedded_gguf_chat_template()
    test_gemma4_handler_rejects_non_gemma_model_architecture()
    test_gemma4_gguf_template_registers_media_tags()
    test_video_mode_requires_image_frames()
    print("advanced llama features: OK")
