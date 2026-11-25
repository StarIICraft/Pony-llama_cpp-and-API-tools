import json
import re
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
LOCALE_ROOT = PLUGIN_ROOT / "locales"


def load_locale(name):
    return json.loads(
        (LOCALE_ROOT / name / "nodeDefs.json").read_text(
            encoding="utf-8"
        )
    )


def test_all_nodes_have_complete_localized_tooltips():
    expected_nodes = {
        "PonyLlamaCppModelLoader",
        "PonyLlamaCppInstruct",
        "PonyLlamaCppParameters",
        "PonyParseJson",
        "PonyJsonToBBox",
        "PonyBBoxToSEGS",
        "PonyBBoxToMask",
        "PonySelectBBox",
        "PonyPromptEnhancerPreset",
        "PonyLlamaCppTextEncoder",
        "PonyApiLlmLoader",
        "PonyApiLlmChain",
        "PonyApiImageEditor",
        "PonyMultiImageResize",
        "PonyMultiImageResizeIO",
        "PonyImageInputBundle",
    }
    generic_zh = re.compile(r"(?:参数。|控制。| 参数)$")
    generic_en = re.compile(
        r"^(?:(?![Cc]onnect\b)[A-Za-z0-9 /()_-]+ output\.|.+ control\.)$"
    )

    for locale_name in ("en", "zh", "zh-CN"):
        data = load_locale(locale_name)
        assert set(data) == expected_nodes
        for node_name, node in data.items():
            assert node.get("display_name")
            assert node.get("description")
            for field_name, entry in node.get("inputs", {}).items():
                tooltip = str(entry.get("tooltip", "")).strip()
                assert tooltip, f"{locale_name}:{node_name}.{field_name}"
                assert not generic_zh.search(tooltip)
                assert not generic_en.search(tooltip)
            for output_name, entry in node.get("outputs", {}).items():
                tooltip = str(entry.get("tooltip", "")).strip()
                assert tooltip, (
                    f"{locale_name}:{node_name}.outputs.{output_name}"
                )


def test_inference_mode_options_are_localized_without_changing_values():
    zh = load_locale("zh-CN")
    en = load_locale("en")

    zh_options = zh["PonyLlamaCppInstruct"]["inputs"][
        "inference_mode"
    ]["options"]
    en_options = en["PonyLlamaCppInstruct"]["inputs"][
        "inference_mode"
    ]["options"]

    assert zh_options["one by one"] == "逐图处理"
    assert zh_options["images"] == "多图合并"
    assert zh_options["video"] == "视频抽帧"
    assert zh_options["text"] == "纯文本"
    assert set(zh_options) == set(en_options) == {
        "one by one",
        "images",
        "video",
        "text",
    }


def test_simplified_and_traditional_chinese_locales_match():
    assert load_locale("zh") == load_locale("zh-CN")


if __name__ == "__main__":
    test_all_nodes_have_complete_localized_tooltips()
    test_inference_mode_options_are_localized_without_changing_values()
    test_simplified_and_traditional_chinese_locales_match()
    print("locale coverage: OK")
