import importlib.util
import sys
import types
from pathlib import Path

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))

PACKAGE_NAME = "pony_binary_megapixels_test"
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


def test_binary_megapixel_constant():
    assert nodes.PIXELS_PER_MEGAPIXEL == 1024 * 1024
    assert nodes.megapixels_to_pixels(1.0) == 1_048_576
    assert nodes.megapixels_to_pixels(0.5) == 524_288


def test_llama_image_resize_exceeds_decimal_one_megapixel():
    image = torch.zeros((2000, 1000, 3), dtype=torch.float32)
    resized = nodes.scale_image(image, max_megapixels=1.0)
    height, width = resized.shape[:2]
    assert width * height <= 1_048_576
    assert width * height > 1_000_000


if __name__ == "__main__":
    test_binary_megapixel_constant()
    test_llama_image_resize_exceeds_decimal_one_megapixel()
    print("binary megapixels: OK")
