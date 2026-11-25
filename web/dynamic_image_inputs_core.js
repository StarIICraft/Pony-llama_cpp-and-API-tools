const DEFAULT_VISIBLE_IMAGE_INPUTS = 3;

function isChineseLocale() {
    const locale = String(
        globalThis?.app?.ui?.settings?.getSettingValue?.("Comfy.Locale")
        || "en"
    ).toLowerCase();
    return locale.startsWith("zh");
}

function localized(value) {
    if (
        value
        && typeof value === "object"
        && !Array.isArray(value)
    ) {
        return isChineseLocale() ? value.zh : value.en;
    }
    return value;
}


export const NODE_CONFIGS = {
    PonyImageInputBundle: {
        pattern: /^image_(\d+)$/,
        format(index) {
            return `image_${index}`;
        },
        label(index) {
            return isChineseLocale()
                ? `图像输入 ${index}`
                : `Image input ${index}`;
        },
        maxInputs: 10,
        type: "IMAGE",
        fixedTailInputs: [],
    },
    PonyMultiImageResizeIO: {
        pattern: /^image_(\d+)$/,
        format(index) {
            return `image_${index}`;
        },
        label(index) {
            return isChineseLocale()
                ? `图像输入 ${index}`
                : `Image input ${index}`;
        },
        maxInputs: 8,
        type: "IMAGE",
        fixedTailInputs: [
            "mask_1",
            "resize_config",
        ],
        imageOutputs: true,
        imageOutputFormat(index) {
            return `图${index} / Image ${index}`;
        },
        imageOutputLabel(index) {
            return isChineseLocale()
                ? `图${index}`
                : `Image ${index}`;
        },
        fixedOutputs: [{
            name: "图1遮罩 / Mask 1",
            label: ["图1遮罩", "Image 1 mask"],
            type: "MASK",
            backendSlot: 8,
        }, {
            name: "图像包 / Image bundle",
            label: ["图像包", "Image bundle"],
            type: "PONY_IMAGE_BUNDLE",
            backendSlot: 9,
        }],
    },
};


export function applyWidgetLabels(node, config) {
    for (const widget of node?.widgets || []) {
        const label = config?.widgetLabels?.[widget?.name];

        if (label) {
            widget.label = localized(label);
        }
    }
}


export function imageInputNumber(name, config) {
    const match = config?.pattern?.exec?.(String(name ?? ""));

    if (!match) {
        return null;
    }

    return Number(match[1] ?? 1);
}


const FIXED_OUTPUT_LABELS = [
    ["输出图像列表", "输出图像列表", "Output image list"],
    ["图1遮罩", "图1遮罩", "Image 1 mask"],
    ["图1拓展遮罩", "图1拓展遮罩", "Image 1 padding mask"],
    ["图1宽度", "图1宽度", "Image 1 width"],
    ["图1高度", "图1高度", "Image 1 height"],
    ["执行报告", "执行报告", "Report"],
    ["多图预览", "多图预览", "Combined preview"],
];


export function outputLabelForName(name, english = false) {
    const text = String(name ?? "");
    const imageMatch = text.match(/^输出图像(\d+)/);

    if (imageMatch) {
        const index = imageMatch[1];
        return english
            ? `Output image ${index}`
            : `输出图像${index}`;
    }

    const fixed = FIXED_OUTPUT_LABELS.find(([prefix]) =>
        text.startsWith(prefix)
    );
    return fixed ? fixed[english ? 2 : 1] : null;
}


function imageEntries(node, config) {
    return (node?.inputs || [])
        .map((input, index) => ({
            input,
            index,
            number: imageInputNumber(input?.name, config),
        }))
        .filter(entry => Number.isInteger(entry.number));
}


function graphLinkById(node, linkId) {
    if (linkId === undefined || linkId === null) {
        return null;
    }

    const links = node?.graph?.links;

    if (!links) {
        return null;
    }

    if (typeof links.get === "function") {
        return (
            links.get(linkId)
            ?? links.get(String(linkId))
            ?? links.get(Number(linkId))
            ?? null
        );
    }

    return links[linkId] ?? links[String(linkId)] ?? null;
}


export function inputLinkForSlot(node, slot) {
    if (
        !node?.graph
        || !Number.isInteger(slot)
        || slot < 0
    ) {
        return null;
    }

    let link = null;

    if (typeof node.getInputLink === "function") {
        try {
            link = node.getInputLink(slot);
        } catch {
            link = null;
        }
    }

    return link ?? graphLinkById(
        node,
        node.inputs?.[slot]?.link,
    );
}


function graphNodeById(graph, id) {
    if (id === undefined || id === null) {
        return null;
    }
    return (
        graph?.getNodeById?.(id)
        ?? graph?.getNodeById?.(String(id))
        ?? graph?.getNodeById?.(Number(id))
        ?? (graph?._nodes || graph?.nodes || []).find(
            entry => String(entry?.id) === String(id)
        )
        ?? null
    );
}


function editableSourceWidget(node, preferredName) {
    const widgets = (node?.widgets || []).filter(
        widget => (
            widget
            && widget.name
            && widget.options?.serialize !== false
        )
    );
    const preferred = widgets.find(
        widget => widget.name === preferredName
    );
    if (preferred) {
        return preferred;
    }
    for (const name of [
        "value",
        "数值",
        "text",
        "text_0",
        "string",
        "Constant",
    ]) {
        const widget = widgets.find(entry => entry.name === name);
        if (widget) {
            return widget;
        }
    }
    return widgets.length === 1 ? widgets[0] : null;
}


function resolveSourceWidgetValue(
    source,
    preferredName,
    visited,
) {
    if (!source) {
        return undefined;
    }
    const key = String(source.id);
    if (visited.has(key)) {
        return undefined;
    }
    visited.add(key);

    if (source.type === "Reroute" || source.type === "SetNode") {
        const next = resolveSourceWidgetValue(
            graphNodeById(
                source.graph,
                inputLinkForSlot(source, 0)?.origin_id,
            ),
            source.inputs?.[0]?.name || preferredName,
            visited,
        );
        if (next !== undefined) {
            return next;
        }
    }

    if (
        source.type === "GetNode"
        && typeof source.getSourceNode === "function"
    ) {
        const next = resolveSourceWidgetValue(
            source.getSourceNode()?.node,
            preferredName,
            visited,
        );
        if (next !== undefined) {
            return next;
        }
    }

    return editableSourceWidget(source, preferredName)?.value;
}


export function resolvedWidgetValue(node, name) {
    const localWidget = (node?.widgets || []).find(
        widget => widget?.name === name
    );
    const inputIndex = (node?.inputs || []).findIndex(
        input => (
            input?.name === name
            || input?.widget?.name === name
        )
    );
    const link = inputLinkForSlot(node, inputIndex);
    if (!link) {
        return localWidget?.value;
    }

    const origin = graphNodeById(
        node.graph,
        link.origin_id ?? link.originId,
    );
    const resolved = resolveSourceWidgetValue(
        origin,
        name,
        new Set(),
    );
    return resolved === undefined
        ? localWidget?.value
        : resolved;
}


export function synchronizedNodeWidth(owner, companion) {
    const ownerWidth = Number(owner?.size?.[0]);
    const targetWidth = (
        Number.isFinite(ownerWidth) && ownerWidth > 0
            ? ownerWidth
            : 1
    );
    let minimumWidth = 1;

    if (typeof companion?.computeSize === "function") {
        const computedWidth = Number(companion.computeSize()?.[0]);
        if (
            Number.isFinite(computedWidth)
            && computedWidth > 0
        ) {
            minimumWidth = computedWidth;
        }
    }

    return Math.max(targetWidth, minimumWidth);
}


function inputLinkForEntry(node, entry) {
    return inputLinkForSlot(node, entry.index);
}


function inputIsLinked(node, entry) {
    return Boolean(inputLinkForEntry(node, entry));
}


function captureInputLinks(node) {
    const links = new Map();

    for (const [slot, input] of (node?.inputs || []).entries()) {
        let link = null;

        if (typeof node.getInputLink === "function") {
            link = node.getInputLink(slot);
        }

        if (!link) {
            link = graphLinkById(node, input?.link);
        }

        if (link) {
            links.set(input, link);
        }
    }

    return links;
}


function graphLinkValues(graph) {
    const links = graph?.links;

    if (!links) {
        return [];
    }

    if (typeof links.values === "function") {
        return [...links.values()];
    }

    return Object.values(links);
}


function beginGraphMutation(graph) {
    graph?.beforeChange?.();
    graph?.canvasAction?.((canvas) => {
        canvas?.emitBeforeChange?.();
    });
}


function endGraphMutation(graph) {
    graph?.canvasAction?.((canvas) => {
        canvas?.emitAfterChange?.();
    });
    graph?.afterChange?.();
}


export function repairInputTargetSlots(node) {
    if (typeof node?.getInputLink === "function") {
        for (
            let slot = 0;
            slot < (node.inputs || []).length;
            slot += 1
        ) {
            const link = node.getInputLink(slot);
            if (
                link
                && Number(link.target_slot) !== slot
            ) {
                link.target_slot = slot;
            }
        }
        return;
    }

    for (const link of graphLinkValues(node?.graph)) {
        if (String(link?.target_id) !== String(node?.id)) {
            continue;
        }

        const index = (node.inputs || []).findIndex(
            input => (
                input?.link !== undefined
                && input.link !== null
                && String(input.link) === String(link.id)
            )
        );

        if (index >= 0) {
            link.target_slot = index;
        }
    }
}


export function repairOutputOriginSlots(node) {
    for (const link of graphLinkValues(node?.graph)) {
        if (String(link?.origin_id) !== String(node?.id)) {
            continue;
        }

        const index = (node.outputs || []).findIndex(
            output => (
                Array.isArray(output?.links)
                && output.links.some(
                    linkedId => String(linkedId) === String(link.id)
                )
            )
        );

        if (index >= 0) {
            link.origin_slot = index;
        }
    }
}


function replaceImageInputLayout(node, config) {
    const capturedLinks = captureInputLinks(node);
    const originalInputs = [...node.inputs];
    let entries = imageEntries(node, config)
        .sort((left, right) => left.number - right.number);
    const connectedLinks = entries.flatMap(entry => {
        const link = capturedLinks.get(entry.input);
        return link ? [link] : [];
    });
    const visibleCount = Math.min(
        config.maxInputs,
        Math.max(
            DEFAULT_VISIBLE_IMAGE_INPUTS,
            connectedLinks.length + 1,
        ),
    );
    const usedNumbers = new Set(
        entries.map(entry => entry.number)
    );

    for (let number = 1; number <= visibleCount; number += 1) {
        if (usedNumbers.has(number)) {
            continue;
        }
        const added = node.addInput(
            config.format(number),
            config.type,
        );
        if (added) {
            added.label = config.label(number);
            added.localized_name = config.label(number);
        }
    }

    entries = imageEntries(node, config)
        .sort((left, right) => left.number - right.number);
    const allImageInputs = new Set(
        entries.map(entry => entry.input)
    );
    const visibleEntries = entries.slice(0, visibleCount);
    const visibleImageInputs = visibleEntries.map(
        entry => entry.input
    );
    const fixedInputs = node.inputs.filter(
        input => !allImageInputs.has(input)
    );
    const firstImageIndex = originalInputs.findIndex(
        input => imageInputNumber(input?.name, config) !== null
    );
    const fixedBeforeCount = (
        firstImageIndex >= 0
            ? originalInputs
                .slice(0, firstImageIndex)
                .filter(input => !allImageInputs.has(input))
                .length
            : Math.max(0, fixedInputs.length - 1)
    );
    const finalInputs = [
        ...fixedInputs.slice(0, fixedBeforeCount),
        ...visibleImageInputs,
        ...fixedInputs.slice(fixedBeforeCount),
    ];
    const finalSlots = new Map(
        finalInputs.map((input, slot) => [input, slot])
    );
    const assignments = new Map(capturedLinks);

    for (const input of allImageInputs) {
        assignments.delete(input);
    }

    connectedLinks.forEach((link, index) => {
        const input = visibleImageInputs[index];
        if (input) {
            assignments.set(input, link);
        }
    });

    for (const [input, link] of assignments) {
        const slot = finalSlots.get(input);
        if (
            Number.isInteger(slot)
            && Number(link.target_slot) !== slot
        ) {
            link.target_slot = slot;
        }
    }

    node.inputs.splice(0, node.inputs.length, ...finalInputs);
    for (const entry of imageEntries(node, config)) {
        const label = config.label(entry.number);
        entry.input.label = label;
        entry.input.localized_name = label;
    }
    node._setConcreteSlots?.();
}


export function compactImageInputs(node, config) {
    if (!node || !Array.isArray(node.inputs) || !config) {
        return;
    }

    const graph = node.graph;
    beginGraphMutation(graph);
    try {
        replaceImageInputLayout(node, config);

        const width = node.size?.[0] || 210;
        const height = node.computeSize?.()[1];

        if (height) {
            node.setSize?.([width, height]);
        }

        node._setConcreteSlots?.();
        repairInputTargetSlots(node);
        repairOutputOriginSlots(node);
        node.arrange?.();
    } finally {
        endGraphMutation(graph);
    }

    node.setDirtyCanvas?.(true, true);
    graph?.setDirtyCanvas?.(true, true);
}


export function normalizeImageInputOrder(node, config) {
    if (
        !node
        || !Array.isArray(node.inputs)
        || !config
    ) {
        return;
    }

    const entries = imageEntries(node, config)
        .sort((left, right) => left.number - right.number);
    const imageInputs = new Set(
        entries.map(entry => entry.input)
    );
    const fixedInputs = node.inputs.filter(
        input => !imageInputs.has(input)
    );
    const firstImageIndex = node.inputs.findIndex(
        input => imageInputs.has(input)
    );
    let insertIndex;

    if (firstImageIndex >= 0) {
        insertIndex = node.inputs
            .slice(0, firstImageIndex)
            .filter(input => !imageInputs.has(input))
            .length;
    } else {
        const tailNames = new Set(config.fixedTailInputs || []);
        insertIndex = fixedInputs.findIndex(
            input => tailNames.has(input?.name)
        );
        if (insertIndex < 0) {
            insertIndex = fixedInputs.length;
        }
    }

    const normalized = [
        ...fixedInputs.slice(0, insertIndex),
        ...entries.map(entry => entry.input),
        ...fixedInputs.slice(insertIndex),
    ];
    const changed = (
        normalized.length !== node.inputs.length
        || normalized.some((input, index) => input !== node.inputs[index])
    );

    if (!changed) {
        return;
    }

    const capturedLinks = captureInputLinks(node);
    const finalSlots = new Map(
        normalized.map((input, slot) => [input, slot])
    );

    for (const [input, link] of capturedLinks) {
        const slot = finalSlots.get(input);
        if (
            Number.isInteger(slot)
            && Number(link.target_slot) !== slot
        ) {
            link.target_slot = slot;
        }
    }

    node.inputs = normalized;
    node._setConcreteSlots?.();
    repairOutputOriginSlots(node);
}


function outputIsLinked(output) {
    return Array.isArray(output?.links)
        ? output.links.length > 0
        : output?.links !== undefined && output.links !== null;
}


function imageOutputNumber(output) {
    for (const value of [
        output?.name,
        output?.localized_name,
        output?.label,
    ]) {
        const match = String(value ?? "")
            .match(
                /^(?:输出图像|图)\s*(\d+)(?:\s*\/\s*(?:Output image|Image)\s*\d+)?$/
            );
        if (match) {
            return Number(match[1]);
        }
    }
    return null;
}


const FIXED_OUTPUT_BACKEND_SLOTS = [
    ["图1遮罩", 8],
    ["图像包", 9],
    ["图1拓展遮罩", 10],
    ["图1宽度", 11],
    ["图1高度", 12],
    ["执行报告", 13],
    ["多图预览", 14],
];


function outputBackendSlot(output, fallback) {
    const number = imageOutputNumber(output);

    if (Number.isInteger(number)) {
        return number - 1;
    }

    const name = String(output?.name ?? "");
    const fixed = FIXED_OUTPUT_BACKEND_SLOTS.find(([prefix]) =>
        name.startsWith(prefix)
    );
    return fixed ? fixed[1] : fallback;
}


function imageOutputEntries(node) {
    return (node?.outputs || [])
        .map((output, index) => ({
            output,
            index,
            number: imageOutputNumber(output),
            linked: outputIsLinked(output),
        }))
        .filter(entry => Number.isInteger(entry.number));
}


function desiredVisibleOutputCount(node, config) {
    const visibleInputs = imageEntries(node, config).length;
    const linkedOutputs = imageOutputEntries(node)
        .filter(entry => entry.linked)
        .map(entry => entry.number);
    const highestLinked = linkedOutputs.length
        ? Math.max(...linkedOutputs)
        : 0;

    return Math.min(
        config.maxInputs,
        Math.max(
            DEFAULT_VISIBLE_IMAGE_INPUTS,
            visibleInputs,
            highestLinked,
        ),
    );
}


function addImageOutput(node, number, config) {
    const insertIndex = number - 1;
    const name = (
        config.imageOutputFormat?.(number)
        || `输出图像${number} / Output image ${number}`
    );
    const label = (
        config.imageOutputLabel?.(number)
        || `输出图像${number}`
    );
    const added = node.addOutput(name, config.type, {
        label,
    }) || node.outputs[node.outputs.length - 1];

    if (!added) {
        return false;
    }

    added.type ||= config.type;
    added.label = label;
    added.localized_name = name;

    const currentIndex = node.outputs.indexOf(added);

    if (currentIndex === -1 || currentIndex === insertIndex) {
        return true;
    }

    const [output] = node.outputs.splice(currentIndex, 1);
    node.outputs.splice(insertIndex, 0, output);

    for (const link of graphLinkValues(node.graph)) {
        if (
            String(link?.origin_id) === String(node.id)
            && Number(link?.origin_slot) >= insertIndex
        ) {
            link.origin_slot += 1;
        }
    }

    node._setConcreteSlots?.();
    assignOutputBackendSlots(node);
    repairOutputOriginSlots(node);
    return true;
}


function addNextImageOutput(node, config) {
    const used = new Set(
        imageOutputEntries(node)
            .map(entry => entry.number)
    );

    for (
        let number = 1;
        number <= config.maxInputs;
        number += 1
    ) {
        if (!used.has(number)) {
            return addImageOutput(node, number, config);
        }
    }

    return false;
}


function assignOutputBackendSlots(node) {
    for (const [index, output] of (node.outputs || []).entries()) {
        output.__dynamicImageBackendSlot = outputBackendSlot(output, index);
    }
    node.__dynamicImageOutputs = true;
}


export function compactImageOutputs(node, config) {
    if (
        !config?.imageOutputs
        || !node
        || !Array.isArray(node.outputs)
    ) {
        return;
    }

    const fixedSpecs = config.fixedOutputs || [];

    function outputTexts(output) {
        return [
            output?.name,
            output?.localized_name,
            output?.label,
        ].map(value => String(value ?? ""));
    }

    function fixedSpecFor(output) {
        const texts = outputTexts(output);
        return fixedSpecs.find(spec => {
            const prefixes = [
                String(spec.name).split(" / ")[0],
                spec.label?.[0],
                spec.label?.[1],
            ].filter(Boolean);
            return texts.some(text =>
                prefixes.some(prefix => text.startsWith(prefix))
            );
        }) || null;
    }

    function mergeOutputLinks(outputs) {
        const links = [];
        for (const output of outputs) {
            for (const value of output?.links || []) {
                if (!links.some(item => (
                    String(item) === String(value)
                ))) {
                    links.push(value);
                }
            }
        }
        return links;
    }

    const linkAssignments = new Map();

    for (const spec of fixedSpecs) {
        if (node.outputs.some(output => fixedSpecFor(output) === spec)) {
            continue;
        }
        node.addOutput(spec.name, spec.type, {
            label: spec.label?.[isChineseLocale() ? 0 : 1],
        });
    }

    const imageGroups = new Map();
    const fixedGroups = new Map();
    const unknownFixed = [];

    for (const output of [...node.outputs]) {
        const number = imageOutputNumber(output);
        if (Number.isInteger(number)) {
            const group = imageGroups.get(number) || [];
            group.push(output);
            imageGroups.set(number, group);
            continue;
        }
        const spec = fixedSpecFor(output);
        if (spec) {
            const group = fixedGroups.get(spec.backendSlot) || [];
            group.push(output);
            fixedGroups.set(spec.backendSlot, group);
        } else {
            unknownFixed.push(output);
        }
    }

    const imageLinks = new Map();
    const fixedLinks = new Map();
    for (const link of graphLinkValues(node.graph)) {
        if (String(link?.origin_id) !== String(node.id)) {
            continue;
        }
        const linkType = String(link.type || "");
        if (linkType === "MASK") {
            const links = fixedLinks.get(8) || [];
            links.push(link.id);
            fixedLinks.set(8, links);
            continue;
        }
        if (linkType === "PONY_IMAGE_BUNDLE") {
            const links = fixedLinks.get(9) || [];
            links.push(link.id);
            fixedLinks.set(9, links);
            continue;
        }
        if (linkType !== "IMAGE") {
            continue;
        }
        const originSlot = Number(link.origin_slot);
        const outputNumber = imageOutputNumber(
            node.outputs?.[originSlot]
        );
        const number = (
            Number.isInteger(outputNumber)
                ? outputNumber
                : originSlot >= 0 && originSlot < 8
                    ? originSlot + 1
                    : null
        );
        if (!Number.isInteger(number)) {
            continue;
        }
        const links = imageLinks.get(number) || [];
        links.push(link.id);
        imageLinks.set(number, links);
    }

    const linkedNumbers = [...imageLinks.keys()];
    const highestLinked = linkedNumbers.length
        ? Math.max(...linkedNumbers)
        : 0;
    const visibleCount = Math.min(
        config.maxInputs,
        Math.max(
            desiredVisibleOutputCount(node, config),
            highestLinked,
        ),
    );

    const canonicalImages = [];
    for (let number = 1; number <= visibleCount; number += 1) {
        let group = imageGroups.get(number) || [];
        if (!group.length) {
            const name = (
                config.imageOutputFormat?.(number)
                || `图${number} / Image ${number}`
            );
            const label = (
                config.imageOutputLabel?.(number)
                || `图${number}`
            );
            const output = node.addOutput(name, config.type, { label });
            output.name = name;
            output.localized_name = name;
            output.label = label;
            output.type = config.type;
            group = [output];
        }
        const preferred = group[0];
        preferred.name = (
            config.imageOutputFormat?.(number)
            || preferred.name
        );
        preferred.localized_name = preferred.name;
        preferred.label = (
            config.imageOutputLabel?.(number)
            || preferred.label
        );
        preferred.type = config.type;
        linkAssignments.set(
            preferred,
            imageLinks.get(number) || [],
        );
        canonicalImages.push(preferred);
    }

    const canonicalFixed = [];
    for (const spec of fixedSpecs) {
        const group = fixedGroups.get(spec.backendSlot) || [];
        let preferred = group[0];
        if (!preferred) {
            preferred = node.addOutput(spec.name, spec.type, {
                label: spec.label?.[isChineseLocale() ? 0 : 1],
            });
        }
        preferred.name = spec.name;
        preferred.localized_name = spec.name;
        preferred.label = (
            spec.label?.[isChineseLocale() ? 0 : 1]
            || preferred.label
        );
        preferred.type = spec.type;
        linkAssignments.set(
            preferred,
            fixedLinks.get(spec.backendSlot) || [],
        );
        canonicalFixed.push(preferred);
    }

    const finalOutputs = [
        ...canonicalImages,
        ...canonicalFixed,
        ...unknownFixed,
    ];
    for (const output of unknownFixed) {
        linkAssignments.set(output, mergeOutputLinks([output]));
    }
    node.outputs.splice(0, node.outputs.length, ...finalOutputs);
    for (const [slot, output] of finalOutputs.entries()) {
        for (const value of linkAssignments.get(output) || []) {
            const link = graphLinkById(node, value);
            if (!link) {
                continue;
            }
            if (Number(link.origin_slot) !== slot) {
                link.origin_slot = slot;
            }
        }
    }

    const width = node.size?.[0] || 210;
    const height = node.computeSize?.()[1];

    if (height) {
        node.setSize?.([width, height]);
    }

    node._setConcreteSlots?.();
    assignOutputBackendSlots(node);
    repairOutputOriginSlots(node);
    node.arrange?.();
    node.setDirtyCanvas?.(true, true);
}


export function remapPromptOutputSlots(prompt, getNodeById) {
    const output = prompt?.output;

    if (!output || typeof output !== "object") {
        return;
    }

    for (const promptNode of Object.values(output)) {
        const inputs = promptNode?.inputs;

        if (!inputs || typeof inputs !== "object") {
            continue;
        }

        for (const value of Object.values(inputs)) {
            if (
                !Array.isArray(value)
                || value.length !== 2
                || !Number.isInteger(Number(value[1]))
            ) {
                continue;
            }

            const source = getNodeById?.(value[0]);

            if (!source?.__dynamicImageOutputs) {
                continue;
            }

            const backendSlot = source.outputs?.[
                Number(value[1])
            ]?.__dynamicImageBackendSlot;

            if (Number.isInteger(backendSlot)) {
                value[1] = backendSlot;
            }
        }
    }
}
