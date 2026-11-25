import importlib.util
import sys
from pathlib import Path

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))
sys.path.insert(0, str(COMFY_ROOT))

spec = importlib.util.spec_from_file_location(
    "multi_image_resize_compat",
    PLUGIN_ROOT / "multi_image_resize.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

controller_type = module.NODE_CLASS_MAPPINGS["PonyMultiImageResize"]
io_type = module.NODE_CLASS_MAPPINGS["PonyMultiImageResizeIO"]


def default_required_inputs():
    values = {}
    for name, (kind, options) in controller_type.INPUT_TYPES()[
        "required"
    ].items():
        values[name] = options.get(
            "default",
            kind[0] if isinstance(kind, list) else None,
        )
    return values


def default_config():
    return default_required_inputs()


def test_controller_exposes_five_fixed_result_outputs():
    assert controller_type.RETURN_TYPES == (
        "MASK",
        "INT",
        "INT",
        "STRING",
        "IMAGE",
    )
    assert len(controller_type.RETURN_NAMES) == 5
    config = default_config()
    assert isinstance(config, dict)
    assert config["divisor"] == 32


def test_megapixel_tooltip_explains_binary_area():
    options = controller_type.INPUT_TYPES()["required"][
        "megapixels"
    ][1]
    assert "1024 × 1024" in options["tooltip"]
    assert "1,048,576" in options["tooltip"]


def test_companion_output_slots_keep_stable_order():
    assert (
        module.NODE_DISPLAY_NAME_MAPPINGS[
            "PonyMultiImageResizeIO"
        ]
        == "动态图像输入输出-小马"
    )
    assert io_type.RETURN_TYPES[:8] == ("IMAGE",) * 8
    assert io_type.RETURN_TYPES[8:] == (
        "MASK",
        "PONY_IMAGE_BUNDLE",
    )
    assert len(io_type.RETURN_TYPES) == 10

    assert io_type.RETURN_NAMES[8].startswith("图1遮罩")
    assert io_type.RETURN_NAMES[9].startswith("图像包")
    assert io_type.RETURN_NAMES[0].startswith("图1 / Image 1")
    assert io_type.RETURN_NAMES[7].startswith("图8 / Image 8")


def test_empty_companion_preserves_output_order():
    node = io_type()
    result = node.resize(
        resize_config=module.json.dumps(default_config())
    )

    assert len(result) == 10
    assert result[8].ndim == 3
    assert isinstance(result[9], module.ResizeBundle)
    assert result[9] == []
    assert result[9].width == 0
    assert result[9].height == 0
    assert isinstance(result[9].report, str)
    assert result[9].preview.ndim == 4


def test_companion_has_eight_images_mask_and_config():
    optional = io_type.INPUT_TYPES()["optional"]
    for index in range(1, 9):
        assert optional[f"image_{index}"][0] == "IMAGE"
    assert optional["image_1"][0] == "IMAGE"
    assert "mask_1" in optional
    assert optional["resize_config"][0] == "STRING"
    assert not any(name.startswith("image_9") for name in optional)


def test_all_controller_widgets_are_runtime_configurable():
    required = set(controller_type.INPUT_TYPES()["required"])
    assert set(module.RUNTIME_CONFIG_FIELDS) == required


def test_linked_widget_values_override_stale_config():
    node = io_type()
    config = default_config()
    config["megapixels"] = 1.0
    config["divisor"] = 0
    image = torch.zeros((1, 600, 2000, 3), dtype=torch.float32)

    result = node.resize(
        image_1=image,
        resize_config=module.json.dumps(config),
        megapixels=3.0,
        divisor=8,
    )

    height, width = result[0].shape[1:3]
    assert (width, height) == (2000, 600)


def test_force_mode_applies_area_multiplier_to_other_images():
    node = io_type()
    config = default_config()
    config["resize_mode"] = module.MODE_FORCE
    config["megapixels"] = 10_000 / module.PIXELS_PER_MEGAPIXEL
    config["area_multiplier"] = 0.5
    config["divisor"] = 0

    result = node.resize(
        image_1=torch.zeros((1, 10, 10, 3)),
        image_2=torch.zeros((1, 8, 8, 3)),
        resize_config=module.json.dumps(config),
    )

    first_height, first_width = result[0].shape[1:3]
    second_height, second_width = result[1].shape[1:3]
    assert first_width * first_height == 10_000
    assert second_width * second_height == 4_900


def test_bundle_preserves_nested_image_order():
    first = object()
    second = object()
    third = object()

    assert module.unpack_image_bundle(
        [first, [second, third]]
    ) == [first, second, third]


def test_bundle_batches_keep_group_order_and_mask_1():
    node = io_type()
    config = default_config()
    config["divisor"] = 0
    first = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
    second = torch.zeros((2, 8, 16, 3), dtype=torch.float32)
    mask = torch.ones((1, 8, 8), dtype=torch.float32)

    result = node.resize(
        image_1=first,
        image_2=second,
        mask_1=mask,
        resize_config=module.json.dumps(config),
    )

    assert result[0].shape[0] == 1
    assert result[1].shape[0] == 2
    assert result[8].shape == (1, 8, 8)
    assert len(result[9]) == 2
    assert result[9].width == 8
    assert result[9].height == 8


def test_controller_extracts_fixed_outputs_from_bundle():
    controller = controller_type()
    images = [torch.zeros((1, 2, 2, 3))]
    mask = torch.zeros((1, 2, 2))
    preview = torch.zeros((1, 2, 2, 3))
    bundle = module.ResizeBundle(
        images,
        mask,
        mask,
        2,
        2,
        "report",
        preview,
    )

    result = controller.extract_results(result_bundle=bundle)

    assert result[0] is mask
    assert result[1:4] == (2, 2, "report")
    assert result[4] is preview


def test_resize_uses_binary_megapixel_area():
    assert module.PIXELS_PER_MEGAPIXEL == 1024 * 1024

    node = io_type()
    config = default_config()
    config["megapixels"] = 1.0
    config["divisor"] = 0

    result = node.resize(
        image_1=torch.zeros((1, 1000, 2000, 3)),
        resize_config=module.json.dumps(config),
    )
    height, width = result[0].shape[1:3]
    assert width * height <= 1_048_576
    assert width * height > 1_000_000


if __name__ == "__main__":
    test_controller_exposes_five_fixed_result_outputs()
    test_megapixel_tooltip_explains_binary_area()
    test_companion_output_slots_keep_stable_order()
    test_empty_companion_preserves_output_order()
    test_companion_has_eight_images_mask_and_config()
    test_all_controller_widgets_are_runtime_configurable()
    test_linked_widget_values_override_stale_config()
    test_force_mode_applies_area_multiplier_to_other_images()
    test_bundle_preserves_nested_image_order()
    test_bundle_batches_keep_group_order_and_mask_1()
    test_controller_extracts_fixed_outputs_from_bundle()
    test_resize_uses_binary_megapixel_area()
    print("multi-image resize compatibility: OK")
