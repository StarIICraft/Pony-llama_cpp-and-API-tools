import importlib.util
import sys
import types
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))

PACKAGE_NAME = "llama_cpp_vlm_retry_test"
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


def completion(text):
    return {"choices": [{"message": {"content": text}}]}


class FakeLlama:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return completion(self.outputs.pop(0))


def run_with_fake(outputs, json_schema=None):
    original = nodes.LLAMA_CPP_STORAGE.llm
    fake = FakeLlama(outputs)
    nodes.LLAMA_CPP_STORAGE.llm = fake
    try:
        result = nodes.llama_cpp_instruct_adv()._create_completion(
            messages=[],
            seed=42,
            parameters={
                "repeat_penalty": 1.0,
                "frequency_penalty": 0.0,
            },
            json_schema=json_schema,
        )
        return result, fake
    finally:
        nodes.LLAMA_CPP_STORAGE.llm = original


def test_degenerate_output_detection():
    node = nodes.llama_cpp_instruct_adv()

    assert node._is_degenerate_output('"  " " " " " " " " " " " " " " " " " " " " "')
    assert node._is_degenerate_output("a a a the the a the the the a a the the a the")
    assert node._is_degenerate_output(
        "<unused23><unused23><unused23><unused23>"
    )
    assert not node._is_degenerate_output("正常的室内商业摄影背景描述。")


def test_retries_once_and_returns_second_output():
    result, fake = run_with_fake([
        '"  " " " " " " " " " " " " " " " " " " " " " " " "',
        "正常的第二段输出",
    ])

    assert result["choices"][0]["message"]["content"] == "正常的第二段输出"
    assert len(fake.calls) == 2
    assert [call["seed"] for call in fake.calls] == [42, 43]
    assert fake.calls[1]["repeat_penalty"] == 1.1
    assert fake.calls[1]["frequency_penalty"] == 0.1


def test_raises_when_retry_also_degrades():
    original = nodes.LLAMA_CPP_STORAGE.llm
    fake = FakeLlama([
        "a a a the the a the the the a a the the a the the",
        "a a a the the a the the the a a the the a the the",
    ])
    nodes.LLAMA_CPP_STORAGE.llm = fake
    try:
        try:
            nodes.llama_cpp_instruct_adv()._create_completion(
                messages=[],
                seed=42,
                parameters={},
            )
        except RuntimeError as error:
            assert "模型输出无效或退化" in str(error)
        else:
            raise AssertionError("expected RuntimeError")
    finally:
        nodes.LLAMA_CPP_STORAGE.llm = original

    assert len(fake.calls) == 2


def test_detects_qwen_image_json_models():
    node = nodes.llama_cpp_instruct_adv()

    assert node._uses_qwen_image_json({
        "model": r"GGUF\qwen3.5_9b_qwen_image_2.1_pe_i2i\model.gguf",
    })
    assert node._uses_qwen_image_json({
        "model": r"GGUF\qwen3.5_9b_qwen_image_2.1_pe_t2i\model.gguf",
    })
    assert node._uses_qwen_image_json({
        "model": "custom_PE_T2I_model.gguf",
    })
    assert node._uses_qwen_image_json({
        "model": "another-i2i-model.gguf",
    })
    assert not node._uses_qwen_image_json({
        "model": r"GGUF\Qwen3-VL-8B\model.gguf",
    })
    assert not node._uses_qwen_image_json({
        "model": "people_model.gguf",
    })


def test_json_mode_extracts_rewritten_prompt():
    schema = nodes.llama_cpp_instruct_adv._qwen_image_json_schema(True)
    result, fake = run_with_fake([
        (
            '{"rewritten_prompt":"高质量室内背景",'
            '"wh_ratio":"","ratio_follow":"<image1>"}'
        ),
    ], json_schema=schema)

    assert result["choices"][0]["message"]["content"] == "高质量室内背景"
    assert "response_format" not in fake.calls[0]


def test_json_mode_retries_invalid_json():
    schema = nodes.llama_cpp_instruct_adv._qwen_image_json_schema(False)
    result, fake = run_with_fake([
        "a a a the the a the the the a a the the a the the",
        '{"rewritten_prompt":"有效背景提示词","wh_ratio":"3:2"}',
    ], json_schema=schema)

    assert result["choices"][0]["message"]["content"] == "有效背景提示词"
    assert len(fake.calls) == 2
    assert "response_format" not in fake.calls[0]
    assert fake.calls[1]["response_format"]["type"] == "json_object"
    assert fake.calls[1]["response_format"]["schema"] == schema


if __name__ == "__main__":
    test_degenerate_output_detection()
    test_retries_once_and_returns_second_output()
    test_raises_when_retry_also_degrades()
    test_detects_qwen_image_json_models()
    test_json_mode_extracts_rewritten_prompt()
    test_json_mode_retries_invalid_json()
    print("llama-cpp degeneration retry: OK")
