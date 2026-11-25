import importlib.util
import sys
import types
from pathlib import Path

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))

import server

server.PromptServer.instance = types.SimpleNamespace(
    routes=types.SimpleNamespace(
        post=lambda *args, **kwargs: (lambda function: function)
    )
)

PACKAGE = "pony_image_bundle_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(PLUGIN_ROOT)]
sys.modules[PACKAGE] = package


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE}.{name}",
        PLUGIN_ROOT / f"{name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bundle_module = load("image_bundle")
nodes_module = load("nodes")
api_module = load("api_nodes")
image_module = load("image_api_nodes")


def test_bundle_keeps_input_order_and_tensor_groups():
    image_1 = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
    image_2 = torch.ones((4, 8, 8, 3), dtype=torch.float32)
    image_3 = torch.full((16, 16, 3), 0.5)

    result = bundle_module.PonyImageInputBundle().bundle(
        image_1=image_1,
        image_2=image_2,
        image_3=image_3,
    )

    assert len(result) == 1
    assert result[0] == [image_1, image_2, image_3]


def test_instruct_collector_accepts_bundle():
    images = [
        torch.zeros((8, 8, 3)),
        torch.ones((2, 8, 8, 3)),
    ]

    result = nodes_module.collect_instruct_images(
        image_bundle=images,
    )

    assert result == images


def test_api_llm_collector_accepts_bundle():
    images = [
        torch.zeros((8, 8, 3)),
        torch.ones((2, 8, 8, 3)),
    ]

    groups = api_module.collect_image_groups(
        image_bundle=images,
    )

    assert len(groups) == 2
    assert len(groups[0]) == 1
    assert len(groups[1]) == 2


def test_image_api_collector_accepts_bundle():
    images = [
        torch.zeros((8, 8, 3)),
        torch.ones((8, 8, 3)),
    ]

    refs, notes = image_module.collect_images(
        {},
        2.5,
        images,
    )

    assert len(refs) == 2
    assert all(value.startswith(b"\x89PNG") for value in refs)
    assert len(notes) == 2


if __name__ == "__main__":
    test_bundle_keeps_input_order_and_tensor_groups()
    test_instruct_collector_accepts_bundle()
    test_api_llm_collector_accepts_bundle()
    test_image_api_collector_accepts_bundle()
    print("image bundle compatibility: OK")
