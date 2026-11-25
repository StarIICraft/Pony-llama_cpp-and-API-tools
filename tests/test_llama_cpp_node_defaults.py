import importlib.util
import sys
import types
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))

PACKAGE_NAME = "llama_cpp_vlm_defaults_test"
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


def test_advanced_node_defaults():
    required = nodes.llama_cpp_instruct_adv.INPUT_TYPES()["required"]

    assert nodes.DEFAULT_MAX_MEGAPIXELS == 0.3
    assert required["max_megapixels"][1]["default"] == 0.3
    assert required["inference_mode"][1]["default"] == "images"
    assert "save_states" not in required
    assert nodes.llama_cpp_instruct_adv.RETURN_TYPES == (
        "STRING",
        "STRING",
    )
    assert nodes.llama_cpp_instruct_adv.RETURN_NAMES == (
        "output",
        "output_list",
    )


def test_parameter_node_defaults():
    required = nodes.llama_cpp_parameters.INPUT_TYPES()["required"]

    assert required["max_tokens"][1]["default"] == 2048
    assert required["temperature"][1]["default"] == 0.3
    assert "state_uid" not in required


def test_removed_nodes_and_persistent_state_are_not_exposed():
    for node_type in (
        "PonyLlamaCppUnload",
        "PonyLlamaCppClearStates",
        "PonyRemoveCodeBlock",
    ):
        assert node_type not in nodes.NODE_CLASS_MAPPINGS
        assert node_type not in nodes.NODE_DISPLAY_NAME_MAPPINGS
    assert not hasattr(nodes.LLAMA_CPP_STORAGE, "messages")
    assert not hasattr(nodes.LLAMA_CPP_STORAGE, "sys_prompts")
    assert not hasattr(nodes.LLAMA_CPP_STORAGE, "clean_state")


def test_prompt_enhancer_uses_synced_preset_files():
    bundled = (
        PLUGIN_ROOT
        / "support"
        / "prompt_enhancer_presets"
    )
    assert bundled.is_dir()
    assert len([
        path
        for path in bundled.iterdir()
        if path.is_file()
    ]) == 19

    names = nodes.get_prompt_enhancer_preset_names()
    assert len(names) == 19
    assert [
        int(name.split("_", 1)[0])
        for name in names
    ] == list(range(1, 20))
    for excluded_name in (
        "LLM模型选择",
        "LLM-api",
        "图像编辑模型选择",
        "fisher-h3-edit-prompt-完整版",
        "fisher-h3-prompt-expander-完整版",
    ):
        assert not any(
            excluded_name in name
            for name in names
        )

    required = nodes.PromptEnhancerPreset.INPUT_TYPES()["required"]
    assert required["preset"][0] == names

    first = names[0]
    text = nodes.PromptEnhancerPreset().main(first)[0]
    assert text
    assert text == nodes.read_prompt_enhancer_preset(first)


if __name__ == "__main__":
    test_advanced_node_defaults()
    test_parameter_node_defaults()
    test_removed_nodes_and_persistent_state_are_not_exposed()
    test_prompt_enhancer_uses_synced_preset_files()
    print("llama-cpp node defaults: OK")
