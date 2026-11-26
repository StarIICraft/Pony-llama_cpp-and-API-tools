import { app } from "../../scripts/app.js";

import {
    NODE_CONFIGS,
    applyWidgetLabels,
    compactImageInputs,
    compactImageOutputs,
    inputLinkForSlot,
    remapPromptOutputSlots,
    synchronizedNodeWidth,
} from "./dynamic_image_inputs_core.js";

globalThis.app ||= app;


const compactTimers = new WeakMap();
const companionTimers = new WeakMap();
const configureSourceIds = new WeakMap();
const resolvedCompanions = new WeakMap();
const MAIN_NODE_TYPES = new Set([
    "PonyLlamaCppInstruct",
    "PonyApiLlmChain",
    "PonyApiImageEditor",
    "PonyMultiImageResize",
]);
const COMPANION_TYPES = {
    PonyMultiImageResize: "PonyMultiImageResizeIO",
};
const BUNDLE_PROPERTY = "imageBundleCompanionId";
const OWNER_PROPERTY = "imageBundleOwnerId";
const COMPANION_VERTICAL_GAP = 2;
const COMPANION_GROUP_PADDING = 8;
const RESIZE_CONFIG_FIELDS = [
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


function graphNodes(graph) {
    return graph?._nodes || graph?.nodes || [];
}


function graphLinks(graph) {
    const links = graph?.links;
    if (!links) {
        return [];
    }
    return typeof links.values === "function"
        ? [...links.values()]
        : Object.values(links);
}


function graphGroups(graph) {
    return graph?.groups || graph?._groups || [];
}


function groupBounding(group) {
    const bounding = (
        group?.boundingRect
        ?? group?._bounding
        ?? group?.bounding
    );
    return bounding && bounding.length >= 4
        ? bounding
        : null;
}


function setGroupBounding(group, values) {
    if (typeof group?.boundingRect?.set === "function") {
        group.boundingRect.set(values);
    } else if (typeof group?._bounding?.set === "function") {
        group._bounding.set(values);
    } else if (Array.isArray(group?.bounding)) {
        group.bounding.splice(0, group.bounding.length, ...values);
    } else if ("bounding" in (group || {})) {
        group.bounding = values;
    } else if ("boundingRect" in (group || {})) {
        group.boundingRect = values;
    } else {
        return false;
    }
    group.updateArea?.();
    group.recomputeInsideNodes?.();
    return true;
}


function nodeVisualBounds(node) {
    const titleHeight = (
        Number(globalThis.LiteGraph?.NODE_TITLE_HEIGHT) || 30
    );
    const x = Number(node?.pos?.[0]) || 0;
    const y = Number(node?.pos?.[1]) || 0;
    return [
        x,
        y - titleHeight,
        Number(node?.size?.[0]) || 1,
        (Number(node?.size?.[1]) || 1) + titleHeight,
    ];
}


function centerInsideBounding(node, bounding) {
    const x = Number(node?.pos?.[0]) || 0;
    const y = Number(node?.pos?.[1]) || 0;
    const width = Number(node?.size?.[0]) || 1;
    const height = Number(node?.size?.[1]) || 1;
    const groupX = Number(bounding[0]);
    const groupY = Number(bounding[1]);
    const groupWidth = Number(bounding[2]);
    const groupHeight = Number(bounding[3]);
    const centerX = x + width / 2;
    const centerY = y + height / 2;
    return (
        groupX <= centerX
        && centerX <= groupX + groupWidth
        && groupY <= centerY
        && centerY <= groupY + groupHeight
    );
}


function expandCompanionGroups(node, companion) {
    const graph = node?.graph;
    const relevantNodes = [
        nodeVisualBounds(node),
        nodeVisualBounds(companion),
    ];
    let changed = false;

    for (const group of graphGroups(graph)) {
        const bounding = groupBounding(group);
        if (!bounding || !centerInsideBounding(node, bounding)) {
            continue;
        }

        const x = Number(bounding[0]);
        const y = Number(bounding[1]);
        const right = x + Number(bounding[2]);
        const bottom = y + Number(bounding[3]);
        let nextX = x;
        let nextY = y;
        let nextRight = right;
        let nextBottom = bottom;

        for (const [nx, ny, width, height] of relevantNodes) {
            nextX = Math.min(
                nextX,
                nx - COMPANION_GROUP_PADDING,
            );
            nextY = Math.min(
                nextY,
                ny - COMPANION_GROUP_PADDING,
            );
            nextRight = Math.max(
                nextRight,
                nx + width + COMPANION_GROUP_PADDING,
            );
            nextBottom = Math.max(
                nextBottom,
                ny + height + COMPANION_GROUP_PADDING,
            );
        }

        const nextBounds = [
            nextX,
            nextY,
            nextRight - nextX,
            nextBottom - nextY,
        ];
        if (nextBounds.some(
            (value, index) => (
                Math.abs(value - Number(bounding[index])) > 0.01
            )
        )) {
            changed = setGroupBounding(group, nextBounds) || changed;
        }
    }

    if (changed) {
        graph?.setDirtyCanvas?.(true, true);
    }
}


function inputSlot(node, name) {
    return (node?.inputs || []).findIndex(
        entry => entry?.name === name
    );
}


function companionTypeFor(node) {
    return (
        COMPANION_TYPES[node?.type]
        || "PonyImageInputBundle"
    );
}


function nodeById(node, id) {
    if (id === undefined || id === null) {
        return null;
    }
    return (
        node?.graph?.getNodeById?.(id)
        ?? node?.graph?.getNodeById?.(Number(id))
        ?? graphNodes(node?.graph).find(
            entry => String(entry?.id) === String(id)
        )
        ?? null
    );
}


function rememberConfigureSource(node, info) {
    const sourceId = info?.id;
    if (sourceId !== undefined && sourceId !== null) {
        configureSourceIds.set(node, String(sourceId));
    }
}


function isMainNodeType(type) {
    return MAIN_NODE_TYPES.has(type);
}


function companionTargetIds(node, companion) {
    const ids = new Set();
    for (const link of graphLinks(node?.graph)) {
        if (
            String(link?.origin_id) !== String(companion?.id)
        ) {
            continue;
        }
        const target = nodeById(node, link.target_id);
        if (target && isMainNodeType(target.type)) {
            ids.add(String(target.id));
        }
    }
    return ids;
}


function companionPositionDistance(node, companion) {
    const titleHeight = (
        Number(globalThis.LiteGraph?.NODE_TITLE_HEIGHT) || 30
    );
    const expectedX = Number(node?.pos?.[0]) || 0;
    const expectedY = (
        (Number(node?.pos?.[1]) || 0)
        - titleHeight
        - (Number(companion?.size?.[1]) || 260)
        - COMPANION_VERTICAL_GAP
    );
    const dx = (
        (Number(companion?.pos?.[0]) || 0) - expectedX
    );
    const dy = (
        (Number(companion?.pos?.[1]) || 0) - expectedY
    );
    return dx * dx + dy * dy;
}


function companionPositionedFor(owner, companion) {
    if (!owner?.pos || !companion?.pos) {
        return false;
    }
    const titleHeight = (
        Number(globalThis.LiteGraph?.NODE_TITLE_HEIGHT) || 30
    );
    const expectedY = (
        (Number(owner.pos?.[1]) || 0)
        - titleHeight
        - (Number(companion.size?.[1]) || 260)
        - COMPANION_VERTICAL_GAP
    );
    return (
        Math.abs(
            (Number(companion.pos?.[0]) || 0)
            - (Number(owner.pos?.[0]) || 0)
        ) <= 8
        && Math.abs(
            (Number(companion.pos?.[1]) || 0)
            - expectedY
        ) <= 12
    );
}


function companionAvailableTo(node, companion, companionType) {
    if (
        !companion
        || companion === node
        || companion.type !== companionType
    ) {
        return false;
    }

    const targets = companionTargetIds(node, companion);
    if (
        [...targets].some(
            targetId => String(targetId) !== String(node.id)
        )
    ) {
        return false;
    }

    const ownerId = companion.properties?.[OWNER_PROPERTY];
    if (ownerId === undefined || ownerId === null) {
        return true;
    }
    if (String(ownerId) === String(node.id)) {
        return true;
    }

    const owner = nodeById(node, ownerId);
    if (!owner || !isMainNodeType(owner.type)) {
        return true;
    }
    if (
        String(owner.properties?.[BUNDLE_PROPERTY])
        === String(companion.id)
    ) {
        return false;
    }
    if (linkedCompanion(owner) === companion) {
        return false;
    }
    return !companionPositionedFor(owner, companion);
}


function closestAvailableCompanion(node, companionType, candidates) {
    return candidates
        .filter(companion => companionAvailableTo(
            node,
            companion,
            companionType,
        ))
        .sort(
            (left, right) => (
                companionPositionDistance(node, left)
                - companionPositionDistance(node, right)
            )
        )[0] || null;
}


function linkedCompanion(node) {
    if (!node?.graph) {
        return null;
    }

    const companionType = companionTypeFor(node);

    const slot = inputSlot(
        node,
        node.type === "PonyMultiImageResize"
            ? "result_bundle"
            : "image_bundle",
    );
    if (slot < 0) {
        return null;
    }

    let link = null;
    link = inputLinkForSlot(node, slot);
    if (!link && node.inputs?.[slot]?.link != null) {
        link = node.graph?.getLink?.(node.inputs[slot].link)
            ?? node.graph?.links?.get?.(node.inputs[slot].link)
            ?? null;
    }
    if (!link) {
        return null;
    }

    const originId = link.origin_id ?? link.originId;
    const origin = nodeById(node, originId);
    return origin?.type === companionType
        ? origin
        : null;
}


function findCompanion(node) {
    if (!node?.graph) {
        return null;
    }

    const companionType = companionTypeFor(node);
    const linked = linkedCompanion(node);
    if (linked) {
        return linked;
    }

    const stored = nodeById(
        node,
        node?.properties?.[BUNDLE_PROPERTY]
    );
    if (
        companionAvailableTo(
            node,
            stored,
            companionType,
        )
    ) {
        return stored;
    }

    const candidates = graphNodes(node?.graph).filter(
        entry => entry?.type === companionType
    );
    const owned = candidates.find(
        entry => (
            String(entry?.properties?.[OWNER_PROPERTY])
            === String(node?.id)
            && companionAvailableTo(
                node,
                entry,
                companionType,
            )
        )
    );
    if (owned) {
        return owned;
    }

    const storedId = node?.properties?.[BUNDLE_PROPERTY];
    if (storedId !== undefined && storedId !== null) {
        const cloned = closestAvailableCompanion(
            node,
            companionType,
            candidates.filter(entry => (
                String(entry.id) !== String(storedId)
                && configureSourceIds.get(entry)
                    === String(storedId)
            )),
        );
        if (cloned) {
            return cloned;
        }
    }

    const sourceId = configureSourceIds.get(node);
    const isClone = (
        sourceId !== undefined
        && String(sourceId) !== String(node?.id)
    );
    if (!isClone && (storedId === undefined || storedId === null)) {
        return null;
    }

    return closestAvailableCompanion(
        node,
        companionType,
        candidates,
    );
}


function moveCompanion(node, companion) {
    if (!node || !companion) {
        return;
    }

    const width = synchronizedNodeWidth(node, companion);
    if (Math.abs(Number(companion.size?.[0]) - width) > 1) {
        companion.setSize?.([
            width,
            companion.size?.[1] || 260,
        ]);
    }

    const height = Number(companion.size?.[1]) || 260;
    const x = Number(node.pos?.[0]) || 0;
    const titleHeight = (
        Number(globalThis.LiteGraph?.NODE_TITLE_HEIGHT) || 30
    );
    const ownerTop = (
        (Number(node.pos?.[1]) || 0) - titleHeight
    );
    const y = (
        ownerTop
        - height
        - COMPANION_VERTICAL_GAP
    );
    if (
        Math.abs(Number(companion.pos?.[0]) - x) > 0.5
        || Math.abs(Number(companion.pos?.[1]) - y) > 0.5
    ) {
        companion.pos = [x, y];
        companion.setDirtyCanvas?.(true, true);
    }

    expandCompanionGroups(node, companion);
    companion.color = node.color;
    companion.bgcolor = node.bgcolor;
    if (companion.mode !== node.mode) {
        companion.mode = node.mode;
        companion.setDirtyCanvas?.(true, true);
    }
}


function ensureCompanion(node) {
    const companionType = companionTypeFor(node);
    if (
        !node?.graph
        || !MAIN_NODE_TYPES.has(node.type)
        || (
            node.type !== "PonyMultiImageResize"
            && inputSlot(node, "image_bundle") < 0
        )
    ) {
        return null;
    }

    let companion = findCompanion(node);

    if (!companion) {
        const litegraph = globalThis.LiteGraph;
        companion = litegraph?.createNode?.(companionType);
        if (!companion) {
            return null;
        }
        node.graph.add?.(companion);
    }

    node.properties ||= {};
    companion.properties ||= {};
    node.properties[BUNDLE_PROPERTY] = companion.id;
    companion.properties[OWNER_PROPERTY] = node.id;
    resolvedCompanions.set(node, companion);

    if (node.type === "PonyMultiImageResize") {
        const slot = inputSlot(node, "result_bundle");
        const link = inputLinkForSlot(node, slot);
        if (
            slot >= 0
            && (!link || String(link.origin_id ?? link.originId)
                !== String(companion.id))
        ) {
            companion.connect?.(9, node, slot);
        }
        compactImageInputs(
            companion,
            NODE_CONFIGS.PonyMultiImageResizeIO,
        );
        const configWidget = (companion.widgets || []).find(
            widget => widget?.name === "resize_config"
        );
        if (configWidget) {
            configWidget.value = "自动同步";
            configWidget.label = "缩放参数（自动同步）";
        }
    } else {
        compactImageInputs(
            companion,
            NODE_CONFIGS.PonyImageInputBundle,
        );

        const slot = inputSlot(node, "image_bundle");
        let link = inputLinkForSlot(node, slot);
        if (!link && node.inputs?.[slot]?.link != null) {
            link = node.graph?.getLink?.(node.inputs[slot].link)
                ?? node.graph?.links?.get?.(node.inputs[slot].link)
                ?? null;
        }
        if (!link) {
            companion.connect?.(0, node, slot);
        }
    }

    moveCompanion(node, companion);
    resolvedCompanions.set(node, companion);
    node.graph.setDirtyCanvas?.(true, true);
    return companion;
}


function scheduleCompanion(node) {
    clearTimeout(companionTimers.get(node));
    const timer = setTimeout(() => {
        if (app.configuringGraph) {
            scheduleCompanion(node);
            return;
        }
        companionTimers.delete(node);
        ensureCompanion(node);
    }, 32);
    companionTimers.set(node, timer);
}


function installCompanionFollowLoop() {
    const canvas = app?.canvas;
    if (!canvas || canvas.__ponyCompanionFollowInstalled) {
        return;
    }
    canvas.__ponyCompanionFollowInstalled = true;
    const original = canvas.onDrawForeground;
    canvas.onDrawForeground = function () {
        const result = original?.apply(this, arguments);
        for (const node of graphNodes(app.graph)) {
            if (!MAIN_NODE_TYPES.has(node?.type)) {
                continue;
            }
            const companion = resolvedCompanions.get(node);
            if (
                companion
                && nodeById(node, companion.id) === companion
            ) {
                moveCompanion(node, companion);
            }
        }
        return result;
    };
}


function scheduleCompact(node, config) {
    clearTimeout(compactTimers.get(node));
    const timer = setTimeout(() => {
        if (app.configuringGraph) {
            scheduleCompact(node, config);
            return;
        }

        compactTimers.delete(node);
        compactImageInputs(node, config);
        compactImageOutputs(node, config);
        const arrange = () => {
            node.arrange?.();
            node.setDirtyCanvas?.(true, true);
            node.graph?.setDirtyCanvas?.(true, true);
        };
        if (typeof requestAnimationFrame === "function") {
            requestAnimationFrame(arrange);
        } else {
            arrange();
        }
    }, 16);
    compactTimers.set(node, timer);
}


function lookupGraphNode(id) {
    const candidates = [
        id,
        String(id).split(":").at(-1),
    ];
    const graphs = [
        app?.graph,
        app?.rootGraph,
    ].filter(Boolean);
    const seen = new Set();

    function search(graph) {
        if (!graph || seen.has(graph)) {
            return null;
        }
        seen.add(graph);

        for (const candidate of candidates) {
            const direct = graph.getNodeById?.(candidate)
                ?? graph.getNodeById?.(Number(candidate));
            if (direct) {
                return direct;
            }
        }

        for (const node of graphNodes(graph)) {
            for (const candidate of candidates) {
                if (String(node?.id) === String(candidate)) {
                    return node;
                }
            }
            if (node?.subgraph) {
                const inner = search(node.subgraph);
                if (inner) {
                    return inner;
                }
            }
        }
        return null;
    }

    for (const graph of graphs) {
        const found = search(graph);
        if (found) {
            return found;
        }
    }
    return null;
}


function eachGraphNode(visitor) {
    const seen = new Set();

    function visit(graph) {
        if (!graph || seen.has(graph)) {
            return;
        }
        seen.add(graph);
        for (const node of graphNodes(graph)) {
            visitor(node);
            if (node?.subgraph) {
                visit(node.subgraph);
            }
        }
    }

    visit(app?.graph);
    if (app?.rootGraph !== app?.graph) {
        visit(app?.rootGraph);
    }
}


function promptNodeById(prompt, id) {
    const output = prompt?.output;
    if (!output || typeof output !== "object") {
        return null;
    }
    const exact = output[String(id)];
    if (exact) {
        return exact;
    }
    return Object.entries(output).find(
        ([key]) => key.split(":").at(-1) === String(id)
    )?.[1] ?? null;
}


function injectResizeConfigs(prompt) {
    eachGraphNode(main => {
        if (main?.type !== "PonyMultiImageResize") {
            return;
        }
        const companionId = main.properties?.[BUNDLE_PROPERTY];
        const companionPrompt = promptNodeById(
            prompt,
            companionId,
        );
        const mainPrompt = promptNodeById(prompt, main.id);
        if (!mainPrompt) {
            return;
        }
        mainPrompt.inputs ||= {};
        mainPrompt.inputs.result_bundle = [
            companionId,
            9,
        ];
        if (!companionPrompt) {
            return;
        }

        companionPrompt.inputs ||= {};
        const config = {};
        for (const widget of main.widgets || []) {
            if (RESIZE_CONFIG_FIELDS.includes(widget?.name)) {
                config[widget.name] = widget.value;
            }
        }
        for (const field of RESIZE_CONFIG_FIELDS) {
            // ComfyUI may inline a primitive value or keep a live link.
            // Both forms override the controller's stale widget value.
            const promptValue = mainPrompt.inputs?.[field];
            if (
                Array.isArray(promptValue)
                && promptValue.length === 2
            ) {
                companionPrompt.inputs[field] = [
                    promptValue[0],
                    promptValue[1],
                ];
            } else if (
                promptValue !== undefined
                && promptValue !== null
            ) {
                config[field] = promptValue;
                delete companionPrompt.inputs[field];
            } else {
                delete companionPrompt.inputs[field];
            }
        }
        const locale = String(
            app?.ui?.settings?.getSettingValue?.("Comfy.Locale")
            || "en"
        ).toLowerCase();
        config.language = locale.startsWith("zh")
            ? "中文"
            : "English";
        companionPrompt.inputs.resize_config = JSON.stringify(config);
    });
}


function installPromptOutputRemap() {
    if (
        typeof app?.graphToPrompt !== "function"
        || app.__ponyDynamicOutputRemapInstalled
    ) {
        return;
    }
    app.__ponyDynamicOutputRemapInstalled = true;
    const originalGraphToPrompt = app.graphToPrompt.bind(app);
    app.graphToPrompt = async function (...args) {
        const prompt = await originalGraphToPrompt(...args);
        injectResizeConfigs(prompt);
        remapPromptOutputSlots(prompt, lookupGraphNode);
        return prompt;
    };
}


function compactConfiguredGraph() {
    eachGraphNode(node => {
        const config = NODE_CONFIGS[
            node?.comfyClass || node?.type
        ];
        if (!config) {
            return;
        }
        compactImageInputs(node, config);
        compactImageOutputs(node, config);
    });
}


function minimizeMainNodeHeights() {
    eachGraphNode(node => {
        if (
            !MAIN_NODE_TYPES.has(node?.comfyClass || node?.type)
            || typeof node?.computeSize !== "function"
        ) {
            return;
        }
        const width = Number(node.size?.[0]) || 320;
        const height = Number(node.computeSize()?.[1]);
        if (
            Number.isFinite(height)
            && height > 0
            && Math.abs(Number(node.size?.[1]) - height) > 1
        ) {
            node.setSize?.([width, height]);
            node.setDirtyCanvas?.(true, true);
        }
    });
}


app.registerExtension({
    name: "PonyToolbox.LlamaDynamicInputs",

    setup() {
        installCompanionFollowLoop();
        installPromptOutputRemap();
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const config = NODE_CONFIGS[nodeData?.name];

        if (MAIN_NODE_TYPES.has(nodeData?.name)) {
            const originalCreated =
                nodeType.prototype.onNodeCreated;
            const originalConfigure =
                nodeType.prototype.onConfigure;
            const originalDrag =
                nodeType.prototype.onDrag;
            const originalDragged =
                nodeType.prototype.onDragged;
            const originalResize =
                nodeType.prototype.onResize;
            const originalConnectionsChange =
                nodeType.prototype.onConnectionsChange;
            const originalRemoved =
                nodeType.prototype.onRemoved;

            nodeType.prototype.onNodeCreated = function () {
                resolvedCompanions.delete(this);
                const result = originalCreated?.apply(
                    this,
                    arguments
                );
                scheduleCompanion(this);
                return result;
            };

            nodeType.prototype.onConfigure = function () {
                rememberConfigureSource(this, arguments[0]);
                resolvedCompanions.delete(this);
                const result = originalConfigure?.apply(
                    this,
                    arguments
                );
                scheduleCompanion(this);
                return result;
            };

            nodeType.prototype.onDrag = function () {
                const result = originalDrag?.apply(
                    this,
                    arguments
                );
                moveCompanion(
                    this,
                    findCompanion(this),
                );
                return result;
            };

            nodeType.prototype.onDragged = function () {
                const result = originalDragged?.apply(
                    this,
                    arguments
                );
                moveCompanion(
                    this,
                    findCompanion(this),
                );
                return result;
            };

            nodeType.prototype.onResize = function () {
                const result = originalResize?.apply(
                    this,
                    arguments
                );
                moveCompanion(
                    this,
                    findCompanion(this),
                );
                return result;
            };

            nodeType.prototype.onConnectionsChange = function () {
                resolvedCompanions.delete(this);
                const result = originalConnectionsChange?.apply(
                    this,
                    arguments
                );
                scheduleCompanion(this);
                return result;
            };

            nodeType.prototype.onRemoved = function () {
                const companion = findCompanion(this);
                resolvedCompanions.delete(this);
                const result = originalRemoved?.apply(
                    this,
                    arguments
                );
                if (
                    companion
                    && String(
                        companion.properties?.[OWNER_PROPERTY]
                    ) === String(this.id)
                ) {
                    this.graph?.remove?.(companion);
                }
                return result;
            };
            return;
        }

        if (!config) {
            return;
        }

        const originalCreated = nodeType.prototype.onNodeCreated;
        const originalConfigure = nodeType.prototype.onConfigure;
        const originalConnectionsChange =
            nodeType.prototype.onConnectionsChange;
        const originalDrag = nodeType.prototype.onDrag;

        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            applyWidgetLabels(this, config);
            scheduleCompact(this, config);
            return result;
        };

        nodeType.prototype.onConfigure = function () {
            rememberConfigureSource(this, arguments[0]);
            const result = originalConfigure?.apply(this, arguments);
            applyWidgetLabels(this, config);
            scheduleCompact(this, config);
            return result;
        };

        nodeType.prototype.onConnectionsChange = function () {
            const result = originalConnectionsChange?.apply(
                this,
                arguments,
            );
            scheduleCompact(this, config);
            return result;
        };

        nodeType.prototype.onDrag = function () {
            const owner = nodeById(
                this,
                this.properties?.[OWNER_PROPERTY],
            );
            if (owner) {
                moveCompanion(owner, this);
            }
            return originalDrag?.apply(this, arguments);
        };
    },

    loadedGraphNode(node) {
        installCompanionFollowLoop();
        if (MAIN_NODE_TYPES.has(node?.comfyClass)) {
            scheduleCompanion(node);
            return;
        }

        const config = NODE_CONFIGS[node?.comfyClass];

        if (config) {
            applyWidgetLabels(node, config);
            scheduleCompact(node, config);
        }
    },

    afterConfigureGraph() {
        compactConfiguredGraph();
        if (typeof requestAnimationFrame === "function") {
            requestAnimationFrame(() => {
                compactConfiguredGraph();
                minimizeMainNodeHeights();
            });
        }
    },
});
