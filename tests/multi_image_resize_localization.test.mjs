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
