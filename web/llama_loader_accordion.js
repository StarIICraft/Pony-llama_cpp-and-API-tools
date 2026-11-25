import { app } from "../../scripts/app.js";


const TARGET_NODE = "PonyLlamaCppModelLoader";
const STATE_PROPERTY = "ponyLoaderAccordion";

const GROUPS = [
    {
        key: "image_tokens",
        fields: ["image_min_tokens", "image_max_tokens"],
        label: {
            zh: "图像 Token 设置",
            en: "Image Token Settings",
        },
    },
    {
        key: "moe_offload",
        fields: ["cpu_moe", "n_cpu_moe"],
        label: {
            zh: "MoE CPU Offload",
            en: "MoE CPU Offload",
        },
    },
];


function isChineseLocale() {
    const locale = String(
        app?.ui?.settings?.getSettingValue?.("Comfy.Locale")
        || "en"
    ).toLowerCase();
    return locale.startsWith("zh");
}


function groupLabel(group) {
    return isChineseLocale() ? group.label.zh : group.label.en;
}


function widgetByName(node, name) {
    return (node?.widgets || []).find(
        widget => widget?.name === name
    );
}


function inputByName(node, name) {
    return (node?.inputs || []).find(
        input => (
            input?.name === name
            || input?.widget?.name === name
        )
    );
}


function groupIsActive(node, group) {
    if (group.key === "image_tokens") {
        return (
            Number(widgetByName(node, "image_min_tokens")?.value || 0) > 0
            || Number(widgetByName(node, "image_max_tokens")?.value || 0) > 0
        );
    }

    return Boolean(widgetByName(node, "cpu_moe")?.value)
        || Number(widgetByName(node, "n_cpu_moe")?.value || 0) > 0;
}


function groupSummary(node, group) {
    if (group.key === "image_tokens") {
        const minimum = Number(
            widgetByName(node, "image_min_tokens")?.value || 0
        );
        const maximum = Number(
            widgetByName(node, "image_max_tokens")?.value || 0
        );
        if (minimum <= 0 && maximum <= 0) {
            return isChineseLocale() ? "自动" : "Auto";
        }
        return `${minimum || "auto"} / ${maximum || "auto"}`;
    }

    const cpuMoe = Boolean(widgetByName(node, "cpu_moe")?.value);
    const nCpuMoe = Number(widgetByName(node, "n_cpu_moe")?.value || 0);
    if (cpuMoe) {
        return isChineseLocale() ? "全部专家" : "All experts";
    }
    if (nCpuMoe > 0) {
        return isChineseLocale() ? `前 ${nCpuMoe} 层` : `First ${nCpuMoe}`;
    }
    return isChineseLocale() ? "关闭" : "Off";
}


function captureWidgetState(widget) {
    if (!widget || widget.__ponyAccordionOriginal) {
        return;
    }
    widget.__ponyAccordionOriginal = {
        computeSize: widget.computeSize,
        draw: widget.draw,
    };
}


function setChildVisible(widget, visible) {
    if (!widget) {
        return;
    }
    captureWidgetState(widget);
    const original = widget.__ponyAccordionOriginal;
    widget.hidden = !visible;

    if (visible) {
        if (original.computeSize) {
            widget.computeSize = original.computeSize;
        } else {
            delete widget.computeSize;
        }
        widget.draw = original.draw;
    } else {
        widget.computeSize = () => [0, -4];
        widget.draw = () => {};
    }
}


function resizeToContent(node) {
    const computed = node?.computeSize?.();
    if (!Array.isArray(computed)) {
        return;
    }
    const width = Number(node.size?.[0]) || Number(computed[0]) || 320;
    const height = Number(computed[1]);
    if (
        Number.isFinite(height)
        && height > 0
        && Math.abs((Number(node.size?.[1]) || 0) - height) > 1
    ) {
        node.setSize?.([width, height]);
    }
}


function updateToggle(node, group, expanded) {
    const state = node.properties?.[STATE_PROPERTY] || {};
    state[group.key] = Boolean(expanded);
    node.properties ||= {};
    node.properties[STATE_PROPERTY] = state;

    for (const name of group.fields) {
        setChildVisible(widgetByName(node, name), expanded);
        const input = inputByName(node, name);
        if (input) {
            input.hidden = !expanded;
        }
    }

    const toggle = (node.widgets || []).find(
        widget => widget?.__ponyAccordionGroup === group.key
    );
    if (toggle) {
        const arrow = expanded ? "▼" : "▶";
        const text = `${arrow} ${groupLabel(group)} · ${groupSummary(node, group)}`;
        toggle.name = text;
        toggle.label = text;
        toggle.value = text;
    }
}


function insertToggleBeforeGroup(node, toggle, group) {
    const widgets = node.widgets || [];
    const currentIndex = widgets.indexOf(toggle);
    if (currentIndex >= 0) {
        widgets.splice(currentIndex, 1);
    }
    const targetIndex = widgets.findIndex(
        widget => widget?.name === group.fields[0]
    );
    widgets.splice(
        targetIndex >= 0 ? targetIndex : widgets.length,
        0,
        toggle,
    );
}


function createToggle(node, group) {
    let toggle = (node.widgets || []).find(
        widget => widget?.__ponyAccordionGroup === group.key
    );
    if (toggle) {
        return toggle;
    }

    const text = `▶ ${groupLabel(group)} · ${groupSummary(node, group)}`;
    toggle = node.addWidget(
        "button",
        text,
        text,
        () => {
            const state = node.properties?.[STATE_PROPERTY] || {};
            const next = !Boolean(state[group.key]);
            updateToggle(node, group, next);
            resizeToContent(node);
            node.setDirtyCanvas?.(true, true);
            app.graph?.setDirtyCanvas?.(true, true);
        },
        { serialize: false },
    );

    toggle.__ponyAccordionGroup = group.key;
    toggle.serialize = false;
    toggle.serializeValue = async () => undefined;
    toggle.options ||= {};
    toggle.options.serialize = false;
    return toggle;
}


function watchChildWidgets(node, group) {
    for (const name of group.fields) {
        const child = widgetByName(node, name);
        if (!child || child.__ponyAccordionWatched) {
            continue;
        }
        const originalCallback = child.callback;
        child.callback = function () {
            const result = originalCallback?.apply(this, arguments);
            const state = node.properties?.[STATE_PROPERTY] || {};
            updateToggle(node, group, Boolean(state[group.key]));
            resizeToContent(node);
            node.setDirtyCanvas?.(true, true);
            return result;
        };
        child.__ponyAccordionWatched = true;
    }
}


export function syncLoaderAccordion(node) {
    if (!node || node.comfyClass !== TARGET_NODE && node.type !== TARGET_NODE) {
        return;
    }

    node.properties ||= {};
    const state = node.properties[STATE_PROPERTY] || {};
    node.properties[STATE_PROPERTY] = state;

    for (const group of GROUPS) {
        const hasChildren = group.fields.some(
            name => Boolean(widgetByName(node, name))
        );
        if (!hasChildren) {
            continue;
        }

        const toggle = createToggle(node, group);
        insertToggleBeforeGroup(node, toggle, group);
        watchChildWidgets(node, group);

        const connected = group.fields.some(
            name => inputByName(node, name)?.link != null
        );
        if (connected) {
            state[group.key] = true;
        } else if (typeof state[group.key] !== "boolean") {
            state[group.key] = connected || groupIsActive(node, group);
        }

        updateToggle(node, group, state[group.key]);
    }

    resizeToContent(node);
}


function scheduleSync(node) {
    syncLoaderAccordion(node);
    if (typeof requestAnimationFrame === "function") {
        requestAnimationFrame(() => syncLoaderAccordion(node));
    }
}


app.registerExtension({
    name: "Pony.Llama.LoaderAccordion",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== TARGET_NODE) {
            return;
        }

        const originalCreated = nodeType.prototype.onNodeCreated;
        const originalConfigure = nodeType.prototype.onConfigure;
        const originalConnectionsChange =
            nodeType.prototype.onConnectionsChange;

        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            scheduleSync(this);
            return result;
        };

        nodeType.prototype.onConfigure = function () {
            const result = originalConfigure?.apply(this, arguments);
            scheduleSync(this);
            return result;
        };

        nodeType.prototype.onConnectionsChange = function () {
            const result = originalConnectionsChange?.apply(
                this,
                arguments,
            );
            scheduleSync(this);
            return result;
        };
    },

    nodeCreated(node) {
        scheduleSync(node);
    },

    loadedGraphNode(node) {
        scheduleSync(node);
    },
});
