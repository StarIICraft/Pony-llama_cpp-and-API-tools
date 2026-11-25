import importlib.util
import sys
import types
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))

PACKAGE_NAME = "pony_image_api_compat_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(PLUGIN_ROOT)]
sys.modules[PACKAGE_NAME] = package

spec = importlib.util.spec_from_file_location(
    f"{PACKAGE_NAME}.image_api_nodes",
    PLUGIN_ROOT / "image_api_nodes.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_escape_wrapped_json_prompt_is_unwrapped():
    raw = (
        '<escape{"rewritten\\_prompt":'
        '"Use \\<image1> and \\<image2>.",'
        '"wh\\_ratio":"","ratio\\_follow":"\\<image1>"}'
    )

    assert module.normalize_prompt_text(raw) == (
        "Use <image1> and <image2>."
    )


def test_plain_escaped_prompt_is_unescaped():
    assert module.normalize_prompt_text(
        r"Use \<image1> with the same subject."
    ) == "Use <image1> with the same subject."


def test_extracts_raw_base64_from_nested_image_key():
    encoded = "iVBORw0KGgo" + ("A" * 120)
    sources, _ = module.extract_content({
        "data": [{"image": encoded}],
    })

    assert sources == [encoded]


def test_extracts_chat_image_url_and_artifacts():
    sources, _ = module.extract_content({
        "output": {
            "content": [{
                "image_url": "https://example.com/result.png",
            }],
        },
        "artifacts": [{
            "base64": "iVBORw0KGgo" + ("B" * 120),
        }],
    })

    assert len(sources) == 2
    assert sources[0] == "https://example.com/result.png"
    assert sources[1].startswith("iVBORw0KGgo")


def test_default_timeout_is_suitable_for_generation():
    options = module.UniversalImageAPINode.INPUT_TYPES()[
        "required"
    ]["请求超时"][1]

    assert options["default"] == 300


def test_image_api_uses_binary_megapixels():
    assert module.PIXELS_PER_MEGAPIXEL == 1024 * 1024
    assert module.megapixels_to_pixels(1.0) == 1_048_576


if __name__ == "__main__":
    test_escape_wrapped_json_prompt_is_unwrapped()
    test_plain_escaped_prompt_is_unescaped()
    test_extracts_raw_base64_from_nested_image_key()
    test_extracts_chat_image_url_and_artifacts()
    test_default_timeout_is_suitable_for_generation()
    test_image_api_uses_binary_megapixels()
    print("image API compatibility: OK")
