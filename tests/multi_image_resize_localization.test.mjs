import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { fileURLToPath } from "node:url";
import path from "node:path";


const TEST_DIRECTORY = path.dirname(fileURLToPath(import.meta.url));
const PLUGIN_DIRECTORY = path.resolve(TEST_DIRECTORY, "..");
const WEB_DIRECTORY = path.join(PLUGIN_DIRECTORY, "web");
let moduleCounter = 0;


function dataModule(source, suffix) {
    return `data:text/javascript;base64,${
        Buffer.from(source).toString("base64")
    }#${suffix}`;
}


async function loadLocalization() {
    let locale = "zh";
    let extension = null;
    globalThis.app = {
        ui: {
            settings: {
                getSettingValue() {
                    return locale;
                },
            },
        },
        registerExtension(value) {
            extension = value;
        },
    };
    const coreSource = await readFile(
        path.join(WEB_DIRECTORY, "dynamic_image_inputs_core.js"),
        "utf8",
    );
    const coreUrl = dataModule(coreSource, "core");
    let source = await readFile(
        path.join(WEB_DIRECTORY, "multi_image_resize.js"),
        "utf8",
    );
    source = source
        .replace(
            /import \{ app \} from "\.\.\/\.\.\/scripts\/app\.js";\r?\n\r?\n/,
            "const app = globalThis.app;\n\n",
        )
        .replace(
            'from "./dynamic_image_inputs_core.js";',
            `from ${JSON.stringify(coreUrl)};`,
        );
    moduleCounter += 1;
    const module = await import(
        dataModule(source, `localization-${moduleCounter}`)
    );
    return {
        localizedOptionLabel: module.localizedOptionLabel,
        extension,
        setLocale(value) {
            locale = value;
        },
    };
}


test("localizes resize combo option labels without changing values", async () => {
    const { localizedOptionLabel } = await loadLocalization();

    assert.equal(
        localizedOptionLabel(
            "统一百万像素限制 / Megapixel limit",
            false,
        ),
        "统一百万像素限制",
    );
    assert.equal(
        localizedOptionLabel(
            "统一百万像素限制 / Megapixel limit",
            true,
        ),
        "Megapixel limit",
    );
    assert.equal(
        localizedOptionLabel(
            "图1手动尺寸/边模式 / Image 1 manual size or edge",
            true,
        ),
        "Image 1 manual size or edge",
    );
    assert.equal(
        localizedOptionLabel("手动", false),
        "手动",
    );
    assert.equal(
        localizedOptionLabel("手动", true),
        "Manual",
    );
    assert.equal(localizedOptionLabel("原图", true), "Original");
    assert.equal(localizedOptionLabel("Lanczos", false), "Lanczos");
});


test("applies localized labels to the combo widget menu", async () => {
    const {
        extension,
        setLocale,
    } = await loadLocalization();
    const mode = "统一百万像素限制 / Megapixel limit";
    const callback = () => {};

    class ResizeNode {
        constructor() {
            this.comfyClass = "PonyMultiImageResize";
            this.type = "PonyMultiImageResize";
            this.properties = {};
            this.inputs = [];
            this.outputs = [];
            this.size = [320, 200];
            this.widgets = [{
                name: "resize_mode",
                type: "combo",
                value: mode,
                callback,
                options: {
                    values: [
                        mode,
                        "所有图像强制缩放到百万像素 / Force megapixel size",
                    ],
                },
            }, {
                name: "area_multiplier",
                type: "number",
                value: 1,
                callback,
            }];
        }

        computeSize() {
            return [320, 200];
        }

        setSize(size) {
            this.size = size;
        }

        setDirtyCanvas() {}
    }

    await extension.beforeRegisterNodeDef(
        ResizeNode,
        { name: "PonyMultiImageResize" },
    );
    const node = new ResizeNode();
    node.onNodeCreated();
    await new Promise(resolve => setTimeout(resolve, 0));

    const widget = node.widgets[0];
    assert.equal(widget.value, mode);
    assert.equal(
        widget.options.getOptionLabel(mode),
        "统一百万像素限制",
    );

    setLocale("en");
    widget.callback();
    assert.equal(widget.value, mode);
    assert.equal(
        widget.options.getOptionLabel(mode),
        "Megapixel limit",
    );

    const areaWidget = node.widgets[1];
    widget.value =
        "所有图像强制缩放到百万像素 / Force megapixel size";
    widget.callback();
    assert.notEqual(areaWidget.type, "hidden");
});


test("copied or template-loaded modes hide all irrelevant controls", async () => {
    const { extension } = await loadLocalization();
    const allWidgets = [
        "resize_mode",
        "megapixels",
        "area_multiplier",
        "color_preset",
        "fill_color",
        "ratio_preset",
        "aspect_ratio",
        "manual_width",
        "manual_height",
        "edge_length",
        "crop_mode",
        "resize_method",
        "interpolation",
        "divisor",
    ];
    const values = {
        resize_mode: "统一百万像素限制 / Megapixel limit",
        megapixels: 2.5,
        area_multiplier: 1,
        color_preset: "灰色 / Gray",
        fill_color: "0,0,0",
        ratio_preset: "原图",
        aspect_ratio: "1:1",
        manual_width: 1024,
        manual_height: 1024,
        edge_length: 2048,
        crop_mode: "填充 / Pad",
        resize_method: "自定义宽高 / Custom width and height",
        interpolation: "Lanczos",
        divisor: 32,
    };
    const cases = [
        {
            name: "megapixel limit",
            values: {},
            visible: [
                "resize_mode",
                "megapixels",
                "area_multiplier",
                "interpolation",
                "divisor",
            ],
        },
        {
            name: "force megapixel",
            values: {
                resize_mode:
                    "所有图像强制缩放到百万像素 / Force megapixel size",
            },
            visible: [
                "resize_mode",
                "megapixels",
                "area_multiplier",
                "interpolation",
                "divisor",
            ],
        },
        {
            name: "manual custom padding",
            values: {
                resize_mode:
                    "图1手动尺寸/边模式 / Image 1 manual size or edge",
                color_preset: "自定义 / Custom",
                resize_method:
                    "自定义宽高 / Custom width and height",
                crop_mode: "填充 / Pad",
            },
            visible: [
                "resize_mode",
                "color_preset",
                "fill_color",
                "manual_width",
                "manual_height",
                "crop_mode",
                "resize_method",
                "interpolation",
                "divisor",
            ],
        },
        {
            name: "manual custom crop",
            values: {
                resize_mode:
                    "图1手动尺寸/边模式 / Image 1 manual size or edge",
                resize_method:
                    "自定义宽高 / Custom width and height",
                crop_mode: "裁剪 / Crop",
            },
            visible: [
                "resize_mode",
                "manual_width",
                "manual_height",
                "crop_mode",
                "resize_method",
                "interpolation",
                "divisor",
            ],
        },
        {
            name: "manual edge",
            values: {
                resize_mode:
                    "图1手动尺寸/边模式 / Image 1 manual size or edge",
                resize_method:
                    "按长边保持比例 / Fit long edge",
            },
            visible: [
                "resize_mode",
                "edge_length",
                "resize_method",
                "interpolation",
                "divisor",
            ],
        },
        {
            name: "aspect-ratio padding",
            values: {
                resize_mode:
                    "统一比例填充 / Pad to aspect ratio",
            },
            visible: [
                "resize_mode",
                "color_preset",
                "ratio_preset",
                "interpolation",
                "divisor",
            ],
        },
        {
            name: "padding custom color and manual ratio",
            values: {
                resize_mode:
                    "统一比例填充 / Pad to aspect ratio",
                color_preset: "自定义 / Custom",
                ratio_preset: "手动",
                aspect_ratio: "16:9",
            },
            visible: [
                "resize_mode",
                "color_preset",
                "fill_color",
                "ratio_preset",
                "aspect_ratio",
                "interpolation",
                "divisor",
            ],
        },
    ];

    class ResizeNode {
        constructor(widgetValues) {
            this.comfyClass = "PonyMultiImageResize";
            this.type = "PonyMultiImageResize";
            this.properties = {};
            this.inputs = [];
            this.outputs = [];
            this.size = [320, 200];
            this.widgets = Object.entries(values).map(
                ([name, value]) => ({
                    name,
                    type: name === "resize_mode" ? "combo" : "number",
                    value: widgetValues[name] ?? value,
                }),
            );
        }

        computeSize() {
            return [320, 200];
        }

        setSize(size) {
            this.size = size;
        }

        setDirtyCanvas() {}
    }

    await extension.beforeRegisterNodeDef(
        ResizeNode,
        { name: "PonyMultiImageResize" },
    );
    for (const testCase of cases) {
        const node = new ResizeNode(testCase.values);
        node.onNodeCreated();
        node.onConfigure({ id: 999 });
        await new Promise(resolve => setTimeout(resolve, 30));

        const widget = name => node.widgets.find(
            item => item.name === name,
        );
        for (const name of allWidgets) {
            if (testCase.visible.includes(name)) {
                assert.notEqual(
                    widget(name).type,
                    "hidden",
                    `${testCase.name}: ${name}`,
                );
            } else {
                assert.equal(
                    widget(name).type,
                    "hidden",
                    `${testCase.name}: ${name}`,
                );
            }
        }
    }
});
