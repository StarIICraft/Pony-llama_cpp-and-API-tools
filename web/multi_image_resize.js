import { app } from "../../scripts/app.js";

import { resolvedWidgetValue } from "./dynamic_image_inputs_core.js";

const NODE_NAME = "PonyMultiImageResize";
const IO_NODE_NAME = "PonyMultiImageResizeIO";

const LABELS = {
    resize_mode: ["缩放模式", "Resize mode"],
    megapixels: ["百万像素上限", "Megapixel limit"],
    area_multiplier: [
        "其余图像根据图1面积乘数",
        "Other images relative to image 1 area",
    ],
    color_preset: ["填充颜色", "Fill color"],
    fill_color: ["自定义RGB颜色", "Custom RGB color"],
    ratio_preset: ["比例预设", "Aspect ratio preset"],
    aspect_ratio: ["手动比例", "Manual aspect ratio"],
    manual_width: ["宽度", "Width"],
    manual_height: ["高度", "Height"],
    edge_length: ["将边缩放到", "Target edge length"],
    crop_mode: ["裁剪模式", "Fit mode"],
    resize_method: ["缩放方式", "Resize method"],
    interpolation: ["缩放插值", "Resize interpolation"],
    divisor: ["整除数", "Divisor"],
    mask_1: ["图1遮罩", "Image 1 mask"],
    resize_config: ["缩放参数（自动同步）", "Resize params (auto)"],
    result_bundle: ["结果包", "Result bundle"],
};

const OUTPUT_LABELS = [
    ["图1", "Image 1"],
    ["图2", "Image 2"],
    ["图3", "Image 3"],
    ["图4", "Image 4"],
    ["图5", "Image 5"],
    ["图6", "Image 6"],
    ["图7", "Image 7"],
    ["图8", "Image 8"],
    ["图1遮罩", "Image 1 mask"],
    ["图像包", "Image bundle"],
];

const CONTROLLER_OUTPUT_LABELS = [
    ["图1拓展遮罩", "Image 1 padding mask"],
    ["图1宽度", "Image 1 width"],
    ["图1高度", "Image 1 height"],
    ["执行报告", "Report"],
    ["多图预览", "Combined preview"],
];

const RESIZE_WIDGET_NAMES = [
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

const COMBO_WIDGET_NAMES = new Set([
    "resize_mode",
    "color_preset",
    "ratio_preset",
    "crop_mode",
    "resize_method",
    "interpolation",
]);

const FIXED_OUTPUT_LABELS = [
    ["图1遮罩", "图1遮罩", "Image 1 mask"],
    ["图像包", "图像包", "Image bundle"],
];

const widgetStates = new WeakMap();
const comboLabelStates = new WeakMap();
const hookedWidgets = new WeakSet();
const initializeTimers = new WeakMap();

function findWidget(node, name) {
    return node.widgets?.find(widget => widget.name === name);
}

function valueOf(node, name) {
    return String(resolvedWidgetValue(node, name) ?? "");
}


export function localizedOptionLabel(value, english) {
    const text = String(value ?? "");
    const parts = text.split(/\s+\/\s+/);
    if (parts.length > 1) {
        return (
            english
                ? parts.slice(1).join(" / ")
                : parts[0]
        ).trim();
    }

    const translated = {
        "手动": ["手动", "Manual"],
        "原图": ["原图", "Original"],
    }[text];
    return translated
        ? translated[english ? 1 : 0]
        : text;
}


function applyComboLocalization(widget, english) {
    if (!widget || !COMBO_WIDGET_NAMES.has(widget.name)) {
        return;
    }
    widget.options ||= {};
    if (!comboLabelStates.has(widget)) {
        comboLabelStates.set(widget, {
            getOptionLabel: widget.options.getOptionLabel,
        });
    }
    const state = comboLabelStates.get(widget);
    widget.options.getOptionLabel = value => {
        const localized = localizedOptionLabel(value, english);
        if (localized !== String(value ?? "")) {
            return localized;
        }
        return state.getOptionLabel
            ? state.getOptionLabel.call(widget.options, value)
            : localized;
    };
}


function syncCompanionConfig(node) {
    const companionId = node?.properties?.imageBundleCompanionId;
    if (
        !node?.graph
        || companionId === undefined
        || companionId === null
    ) {
        return;
    }
    const companion = (
        node?.graph?.getNodeById?.(companionId)
        ?? node?.graph?.getNodeById?.(Number(companionId))
    );
    const configWidget = findWidget(companion, "resize_config");
    if (!configWidget) {
        return;
    }
    configWidget.value = "自动同步";
}


function outputLabelFor(output, index, english) {
    const name = String(output?.name ?? "");
    const image = /^图(\d+)(?:\s*\/\s*Image\s*\d+)?$/.exec(name);

    if (image) {
        return english ? `Image ${image[1]}` : `图${image[1]}`;
    }

    const fixed = FIXED_OUTPUT_LABELS.find(([prefix]) =>
        name.startsWith(prefix)
    );
    if (fixed) {
        return fixed[english ? 2 : 1];
    }

    const backendSlot = Number(output?.__dynamicImageBackendSlot);
    if (
        Number.isInteger(backendSlot)
        && OUTPUT_LABELS[backendSlot]
    ) {
        return OUTPUT_LABELS[backendSlot][english ? 1 : 0];
    }

    return OUTPUT_LABELS[index]?.[english ? 1 : 0];
}

function setVisible(widget, visible) {
    if (!widget) return;

    // 不干预已经转换为输入端口的控件。
    if (String(widget.type).startsWith("converted-widget")) return;

    if (!widgetStates.has(widget)) {
        widgetStates.set(widget, {
            type: widget.type,
            computeSize: widget.computeSize,
        });
    }

    const original = widgetStates.get(widget);

    if (visible) {
        widget.type = original.type;
        if (original.computeSize === undefined) {
            delete widget.computeSize;
        } else {
            widget.computeSize = original.computeSize;
        }
    } else {
        widget.type = "hidden";
        widget.computeSize = () => [0, -4];
    }
}

function updateNode(node) {
    const nodeType = node.comfyClass || node.type;
    const isController = nodeType === NODE_NAME;
    if (isController) {
        syncCompanionConfig(node);
    }
    const english = !String(
        app?.ui?.settings?.getSettingValue?.("Comfy.Locale")
        || "en"
    ).toLowerCase().startsWith("zh");
    const languageColumn = english ? 1 : 0;

    for (const widget of node.widgets ?? []) {
        if (LABELS[widget.name]) {
            widget.label = LABELS[widget.name][languageColumn];
        }
        applyComboLocalization(widget, english);
    }

    for (const input of node.inputs ?? []) {
        const match = /^image_(\d+)$/.exec(input.name);

        if (match) {
            input.label = english
                ? `Image ${match[1]}`
                : `图${match[1]}`;
        } else if (LABELS[input.name]) {
            input.label = LABELS[input.name][languageColumn];
        }
    }

    const outputLabels = (
        isController
            ? CONTROLLER_OUTPUT_LABELS
            : OUTPUT_LABELS
    );
    for (let index = 0; index < (node.outputs?.length ?? 0); index++) {
        const label = (
            isController
                ? outputLabels[index]?.[languageColumn]
                : outputLabelFor(
                    node.outputs[index],
                    index,
                    english,
                )
        );
        if (label) {
            node.outputs[index].label = label;
        }
    }

    if (!isController) {
        setVisible(findWidget(node, "resize_config"), false);
        node.setDirtyCanvas?.(true, true);
        return;
    }

    const mode = valueOf(node, "resize_mode");
    const limit = mode.startsWith("统一百万像素限制");
    const force = mode.startsWith("所有图像强制缩放");
    const manual = mode.startsWith("图1手动尺寸/边模式");
    const padding = mode.startsWith("统一比例填充");

    const resizeMethod = valueOf(node, "resize_method");
    const customSize = resizeMethod.startsWith("自定义宽高");
    const cropMode = valueOf(node, "crop_mode");
    const fitPadding = cropMode.startsWith("填充");

    const needsFillColor = padding || (manual && customSize && fitPadding);
    const isCustomColor = valueOf(node, "color_preset").startsWith("自定义");
    const manualRatio = valueOf(node, "ratio_preset") === "手动";

    const visibility = {
        megapixels: limit || force,
        area_multiplier: limit || force,
        color_preset: needsFillColor,
        fill_color: needsFillColor && isCustomColor,
        ratio_preset: padding,
        aspect_ratio: padding && manualRatio,
        manual_width: manual && customSize,
        manual_height: manual && customSize,
        edge_length: manual && !customSize,
        crop_mode: manual && customSize,
        resize_method: manual,
        interpolation: true,
        divisor: true,
    };

    // 处理 unused 变量，同时保留模式判定易读性。
    void force;

    for (const [name, visible] of Object.entries(visibility)) {
        setVisible(findWidget(node, name), visible);
    }

    if (typeof node.computeSize === "function") {
        const computed = node.computeSize();
        node.setSize([
            Math.max(node.size?.[0] ?? 0, computed[0]),
            computed[1],
        ]);
    }

    node.setDirtyCanvas(true, true);
}

function hookWidget(node, name) {
    const widget = findWidget(node, name);
    if (!widget || hookedWidgets.has(widget)) return;

    hookedWidgets.add(widget);
    const previous = widget.callback;

    widget.callback = function (...args) {
        try {
            return previous?.apply(this, args);
        } finally {
            updateNode(node);
        }
    };
}

function initialize(node) {
    const nodeType = node.comfyClass || node.type;
    if (nodeType !== NODE_NAME) {
        updateNode(node);
        return;
    }
    for (const name of RESIZE_WIDGET_NAMES) {
        hookWidget(node, name);
    }

    updateNode(node);
}


function scheduleInitialize(node) {
    clearTimeout(initializeTimers.get(node));
    const timer = setTimeout(() => {
        if (app.configuringGraph) {
            scheduleInitialize(node);
            return;
        }
        initializeTimers.delete(node);
        initialize(node);
        if (typeof requestAnimationFrame === "function") {
            requestAnimationFrame(() => {
                if (!app.configuringGraph) {
                    initialize(node);
                }
            });
        }
    }, 0);
    initializeTimers.set(node, timer);
}


app.registerExtension({
    name: "PonyToolbox.MultiImageResize.UI",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (![NODE_NAME, IO_NODE_NAME].includes(nodeData.name)) return;

        const previousCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = previousCreated?.apply(this, args);
            scheduleInitialize(this);
            return result;
        };

        const previousConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = previousConfigure?.apply(this, args);
            scheduleInitialize(this);
            return result;
        };

        const previousAdded = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function (...args) {
            const result = previousAdded?.apply(this, args);
            scheduleInitialize(this);
            return result;
        };
    },

    loadedGraphNode(node) {
        if ([NODE_NAME, IO_NODE_NAME].includes(node?.comfyClass)) {
            scheduleInitialize(node);
        }
    },
});
