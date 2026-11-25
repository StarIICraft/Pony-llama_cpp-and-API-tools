import assert from "node:assert/strict";
import test from "node:test";

import {
    NODE_CONFIGS,
    compactImageInputs,
    compactImageOutputs,
    inputLinkForSlot,
    normalizeImageInputOrder,
    outputLabelForName,
    repairInputTargetSlots,
    remapPromptOutputSlots,
    resolvedWidgetValue,
    synchronizedNodeWidth,
} from "../web/dynamic_image_inputs_core.js";

globalThis.app = {
    ui: {
        settings: {
            getSettingValue() {
                return "zh";
            },
        },
    },
};


function imageNames(node, config) {
    return node.inputs
        .map(input => input.name)
        .filter(name => config.pattern.test(name));
}


function tailNames(node, config) {
    return node.inputs
        .map(input => input.name)
        .filter(name => !config.pattern.test(name));
}


function fakeNode(names, linkedNumbers = []) {
    const linked = new Set(linkedNumbers);
    const links = new Map();
    const node = {
        id: 10,
        size: [300, 100],
        inputs: [],
        outputs: [],
        graph: { links },
        getInputLink(slot) {
            return [...links.values()].find(
                link => (
                    String(link.target_id) === String(this.id)
                    && link.target_slot === slot
                )
            ) || null;
        },
        addInput(name, type) {
            const input = { name, type, link: null };
            this.inputs.push(input);
            return input;
        },
        removeInput(index) {
            const [removed] = this.inputs.splice(index, 1);
            for (const link of links.values()) {
                if (link.target_slot > index) {
                    link.target_slot -= 1;
                }
            }
            if (removed?.link !== null) {
                links.delete(removed.link);
            }
            return removed;
        },
        addOutput(name, type, options = {}) {
            const output = {
                name,
                type,
                label: options.label,
                links: [],
            };
            this.outputs.push(output);
            return output;
        },
        removeOutput(index) {
            const [removed] = this.outputs.splice(index, 1);
            for (const link of links.values()) {
                if (
                    String(link.origin_id) === String(this.id)
                    && Number(link.origin_slot) > index
                ) {
                    link.origin_slot -= 1;
                }
            }
            return removed;
        },
        computeSize() {
            return [300, this.inputs.length * 20 + 80];
        },
        setSize(size) {
            this.size = size;
        },
        setDirtyCanvas() {},
        _setConcreteSlots() {},
    };

    for (const [index, name] of names.entries()) {
        const number = Number(
            String(name).match(/(?:images_|图片_|参考图_)?(\d+)$/)?.[1] ?? 1
        );
        const isLinked = linked.has(number);
        const link = isLinked ? `link-${index}` : null;
        node.inputs.push({ name, type: "IMAGE", link });
        if (link) {
            links.set(link, {
                id: link,
                target_id: node.id,
                target_slot: index,
            });
        }
    }

    return node;
}


function allInputNames(config, count) {
    return Array.from({ length: count }, (_, index) =>
        config.format(index + 1)
    );
}


test("resolves a linked primitive widget value for UI state", () => {
    const graph = {
        links: new Map(),
        _nodes: [],
    };
    const source = {
        id: 20,
        type: "easy float",
        graph,
        widgets: [{ name: "value", value: 3.04 }],
    };
    const main = {
        id: 10,
        type: "PonyMultiImageResize",
        graph,
        widgets: [
            { name: "megapixels", value: 1.0 },
        ],
        inputs: [{
            name: "megapixels",
            widget: { name: "megapixels" },
            link: 1,
        }],
    };
    graph._nodes.push(main, source);
    graph.links.set(1, {
        id: 1,
        origin_id: 20,
        origin_slot: 0,
        target_id: 10,
        target_slot: 0,
        type: "FLOAT",
    });

    assert.equal(
        resolvedWidgetValue(main, "megapixels"),
        3.04,
    );
    source.widgets[0].value = 5.0;
    assert.equal(
        resolvedWidgetValue(main, "megapixels"),
        5.0,
    );
});


test("keeps the first three image inputs visible", () => {
    const config = NODE_CONFIGS.PonyImageInputBundle;
    const node = fakeNode(allInputNames(config, 6));

    compactImageInputs(node, config);

    assert.deepEqual(imageNames(node, config), [
        "image_1",
        "image_2",
        "image_3",
    ]);
});


test("input lookup is safe before a node is attached to a graph", () => {
    const node = {
        inputs: [{ name: "image_1", type: "IMAGE", link: 7 }],
        getInputLink() {
            throw new Error("NullGraphError");
        },
    };

    assert.equal(inputLinkForSlot(node, 0), null);
});


test("input lookup reads the graph link store when available", () => {
    const link = {
        id: 7,
        target_id: 10,
        target_slot: 0,
    };
    const node = {
        graph: {
            links: new Map([[7, link]]),
        },
        inputs: [{ name: "image_1", type: "IMAGE", link: 7 }],
        getInputLink() {
            throw new Error("temporary lookup failure");
        },
    };

    assert.equal(inputLinkForSlot(node, 0), link);
});


test("companion width exactly follows the owner above its shrink limit", () => {
    const companion = {
        computeSize() {
            return [180, 140];
        },
    };

    assert.equal(
        synchronizedNodeWidth({ size: [520, 300] }, companion),
        520,
    );
    assert.equal(
        synchronizedNodeWidth({ size: [160, 300] }, companion),
        180,
    );
});


test("fills all missing default slots from a single image input", () => {
    const config = NODE_CONFIGS.PonyImageInputBundle;
    const node = fakeNode(["image_1"]);

    compactImageInputs(node, config);

    assert.deepEqual(imageNames(node, config), [
        "image_1",
        "image_2",
        "image_3",
    ]);
});


test("preserves connected inputs beyond the default three", () => {
    const config = NODE_CONFIGS.PonyImageInputBundle;
    const node = fakeNode(allInputNames(config, 10), [1, 2, 3, 4, 5]);

    compactImageInputs(node, config);

    assert.deepEqual(imageNames(node, config), [
        "image_1",
        "image_2",
        "image_3",
        "image_4",
        "image_5",
        "image_6",
    ]);
});


test("fills multiple gaps before trailing non-image inputs", () => {
    const config = NODE_CONFIGS.PonyImageInputBundle;
    const node = fakeNode(
        ["image_1", "image_2", "image_5", "queue_handler"],
        [1, 2, 5, 9],
    );

    compactImageInputs(node, config);

    assert.deepEqual(imageNames(node, config), [
        "image_1",
        "image_2",
        "image_3",
        "image_4",
    ]);
    assert.deepEqual(tailNames(node, config), ["queue_handler"]);
});


test("keeps image inputs before queue and skill", () => {
    const config = NODE_CONFIGS.PonyImageInputBundle;
    const node = fakeNode(
        ["image_1", "image_2", "image_3", "image_4", "queue_handler", "skill"],
        [1, 2, 3, 4, 5],
    );
    const queueLink = node.inputs[4].link;
    const skillLink = node.inputs[5].link;

    compactImageInputs(node, config);

    assert.deepEqual(imageNames(node, config), [
        "image_1",
        "image_2",
        "image_3",
        "image_4",
        "image_5",
    ]);
    assert.deepEqual(tailNames(node, config), [
        "queue_handler",
        "skill",
    ]);
    assert.equal(node.graph.links.get(queueLink).target_slot, 5);
    assert.equal(node.graph.links.get(skillLink).target_slot, 6);
    assert.equal(node.inputs.at(-1).name, "skill");
});


test("stops at the unified ten image input limit", () => {
    const config = NODE_CONFIGS.PonyImageInputBundle;
    const node = fakeNode(
        allInputNames(config, 12),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    );

    compactImageInputs(node, config);

    assert.equal(imageNames(node, config).length, 10);
});


test("multi-resize companion keeps mask and config after eight images", () => {
    const config = NODE_CONFIGS.PonyMultiImageResizeIO;
    const node = fakeNode(
        [
            ...allInputNames(config, 10),
            "mask_1",
            "resize_config",
        ],
        [1, 2, 3, 4, 5, 6, 7, 8],
    );

    compactImageInputs(node, config);

    assert.equal(imageNames(node, config).length, 8);
    assert.deepEqual(
        node.inputs.slice(-2).map(input => input.name),
        ["mask_1", "resize_config"],
    );
});


test("multi-resize image edits never reorder companion outputs", () => {
    const config = NODE_CONFIGS.PonyMultiImageResizeIO;
    const node = fakeNode(
        [
            ...allInputNames(config, 8),
            "mask_1",
            "resize_config",
        ],
        [1, 2, 4, 5],
    );
    node.outputs = Array.from({ length: 10 }, (_, index) => ({
        name: `output_${index}`,
        type: index < 8 ? "IMAGE" : "MASK",
        links: [],
    }));
    const outputsBefore = structuredClone(node.outputs);

    compactImageInputs(node, config);

    assert.deepEqual(node.outputs, outputsBefore);
    assert.deepEqual(
        node.inputs.slice(-2).map(input => input.name),
        ["mask_1", "resize_config"],
    );
});


test("multi-resize image outputs follow inputs while fixed outputs stay aligned", () => {
    const config = NODE_CONFIGS.PonyMultiImageResizeIO;
    const node = fakeNode(
        [
            ...allInputNames(config, 8),
            "mask_1",
            "resize_config",
        ],
        [1, 2, 3],
    );
    const outputNames = [
        ...Array.from(
            { length: 8 },
            (_, index) => `图${index + 1} / Image ${index + 1}`,
        ),
        "图1遮罩 / Mask 1",
        "图像包 / Image bundle",
    ];
    node.outputs = outputNames.map((name, index) => ({
        name,
        type: index < 8 ? "IMAGE" : "MASK",
        links: [],
    }));
    const maskLink = {
        id: "mask-output",
        origin_id: node.id,
        origin_slot: 8,
        target_id: 20,
        target_slot: 0,
        type: "MASK",
    };
    node.outputs[8].links.push(maskLink.id);
    node.graph.links.set(maskLink.id, maskLink);

    compactImageInputs(node, config);
    compactImageOutputs(node, config);

    assert.deepEqual(
        node.outputs.slice(0, 4).map(output => output.name),
        [
            "图1 / Image 1",
            "图2 / Image 2",
            "图3 / Image 3",
            "图4 / Image 4",
        ],
    );
    assert.equal(node.outputs[4].name, "图1遮罩 / Mask 1");
    assert.equal(node.outputs[5].name, "图像包 / Image bundle");
    assert.equal(maskLink.origin_slot, 4);
    assert.equal(node.outputs[4].__dynamicImageBackendSlot, 8);
    assert.equal(node.outputs[5].__dynamicImageBackendSlot, 9);
    assert.equal(node.outputs.at(-1).__dynamicImageBackendSlot, 9);

    const fourth = node.inputs.find(input => input.name === "image_4");
    fourth.link = "image-4";
    node.graph.links.set(fourth.link, {
        id: fourth.link,
        target_id: node.id,
        target_slot: node.inputs.indexOf(fourth),
    });
    compactImageInputs(node, config);
    compactImageOutputs(node, config);
    assert.equal(
        node.outputs.filter(output => /^图\d+ \/ Image \d+$/.test(output.name)).length,
        5,
    );

    fourth.link = null;
    node.graph.links.delete("image-4");
    compactImageInputs(node, config);
    compactImageOutputs(node, config);
    assert.equal(
        node.outputs.filter(output => /^图\d+ \/ Image \d+$/.test(output.name)).length,
        4,
    );
    assert.equal(maskLink.origin_slot, 4);
});


test("prompt remap converts visible output slots to backend slots", () => {
    const config = NODE_CONFIGS.PonyMultiImageResizeIO;
    const node = fakeNode(
        [
            ...allInputNames(config, 8),
            "mask_1",
            "resize_config",
        ],
        [1, 2, 3],
    );
    node.outputs = [
        ...Array.from({ length: 4 }, (_, index) => ({
            name: `图${index + 1} / Image ${index + 1}`,
            type: "IMAGE",
            links: [],
        })),
        ...["图1遮罩 / Mask 1", "图像包 / Image bundle"]
            .map(name => ({ name, type: "MASK", links: [] })),
    ];
    compactImageInputs(node, config);
    compactImageOutputs(node, config);

    const prompt = {
        output: {
            20: {
                inputs: {
                    image: [node.id, 3],
                    mask: [node.id, 4],
                    bundle: [node.id, 5],
                },
            },
        },
    };
    remapPromptOutputSlots(
        prompt,
        id => String(id) === String(node.id) ? node : null,
    );

    assert.equal(prompt.output[20].inputs.image[1], 3);
    assert.equal(prompt.output[20].inputs.mask[1], 8);
    assert.equal(prompt.output[20].inputs.bundle[1], 9);
});


test("reloaded duplicate outputs are merged without shifting fixed outputs", () => {
    const config = NODE_CONFIGS.PonyMultiImageResizeIO;
    const node = fakeNode(
        [
            ...allInputNames(config, 8),
            "mask_1",
            "resize_config",
        ],
        [1, 2],
    );
    const link = (id, slot, type) => ({
        id,
        origin_id: node.id,
        origin_slot: slot,
        target_id: 20,
        target_slot: 0,
        type,
    });
    const output = (name, slot, links = [], type = "IMAGE") => {
        for (const id of links) {
            node.graph.links.set(id, link(id, slot, type));
        }
        return { name, type, links };
    };
    node.outputs = [
        output("图1 / Image 1", 0, ["image-1a"]),
        output("图2 / Image 2", 1, ["image-2a"]),
        output("图1遮罩 / Mask 1", 2, ["mask-a"], "MASK"),
        output("图像包 / Image bundle", 3, ["bundle-a"], "PONY_IMAGE_BUNDLE"),
        output("图1 / Image 1", 4, ["image-1b"]),
        output("图2 / Image 2", 5, ["image-2b"]),
        output("图1遮罩 / Mask 1", 6, ["mask-b"], "MASK"),
        output("图像包 / Image bundle", 7, ["bundle-b"], "PONY_IMAGE_BUNDLE"),
        output("图8 / Image 8", 8),
    ];

    compactImageInputs(node, config);
    compactImageOutputs(node, config);

    assert.deepEqual(
        node.outputs.map(item => item.name),
        [
            "图1 / Image 1",
            "图2 / Image 2",
            "图3 / Image 3",
            "图1遮罩 / Mask 1",
            "图像包 / Image bundle",
        ],
    );
    const linksAt = slot => [...node.graph.links.values()]
        .filter(link => (
            String(link.origin_id) === String(node.id)
            && link.origin_slot === slot
        ))
        .map(link => link.id)
        .sort();
    assert.deepEqual(linksAt(0), ["image-1a", "image-1b"]);
    assert.deepEqual(linksAt(1), ["image-2a", "image-2b"]);
    assert.deepEqual(linksAt(3), ["mask-a", "mask-b"]);
    assert.deepEqual(linksAt(4), ["bundle-a", "bundle-b"]);
    assert.deepEqual(
        node.outputs.map(item => item.__dynamicImageBackendSlot),
        [0, 1, 2, 8, 9],
    );

    compactImageOutputs(node, config);
    assert.equal(node.outputs.length, 5);
});


test("mask and bundle links recover from stale image output slots", () => {
    const config = NODE_CONFIGS.PonyMultiImageResizeIO;
    const node = fakeNode(
        [
            ...allInputNames(config, 8),
            "mask_1",
            "resize_config",
        ],
        [1, 2, 3, 4],
    );
    node.outputs = [
        ...Array.from({ length: 7 }, (_, index) => ({
            name: `图${index + 1} / Image ${index + 1}`,
            type: "IMAGE",
            links: [],
        })),
        { name: "图1遮罩 / Mask 1", type: "MASK", links: [] },
        { name: "图像包 / Image bundle", type: "PONY_IMAGE_BUNDLE", links: [] },
    ];
    node.graph.links.set("image-stale", {
        id: "image-stale",
        origin_id: node.id,
        origin_slot: 0,
        target_id: 20,
        target_slot: 0,
        type: "IMAGE",
    });
    node.graph.links.set("mask-stale", {
        id: "mask-stale",
        origin_id: node.id,
        origin_slot: 6,
        target_id: 20,
        target_slot: 1,
        type: "MASK",
    });
    node.graph.links.set("bundle-stale", {
        id: "bundle-stale",
        origin_id: node.id,
        origin_slot: 7,
        target_id: 20,
        target_slot: 2,
        type: "PONY_IMAGE_BUNDLE",
    });

    compactImageInputs(node, config);
    compactImageOutputs(node, config);

    assert.deepEqual(
        node.outputs.map(output => output.name),
        [
            "图1 / Image 1",
            "图2 / Image 2",
            "图3 / Image 3",
            "图4 / Image 4",
            "图5 / Image 5",
            "图1遮罩 / Mask 1",
            "图像包 / Image bundle",
        ],
    );
    assert.equal(node.graph.links.get("image-stale").origin_slot, 0);
    assert.equal(node.graph.links.get("mask-stale").origin_slot, 5);
    assert.equal(node.graph.links.get("bundle-stale").origin_slot, 6);
});


test("shows exactly one new image slot after the highest connection", () => {
    const config = NODE_CONFIGS.PonyImageInputBundle;
    const node = fakeNode(
        allInputNames(config, 6),
        [1, 2, 3, 4],
    );

    compactImageInputs(node, config);

    assert.deepEqual(imageNames(node, config), [
        "image_1",
        "image_2",
        "image_3",
        "image_4",
        "image_5",
    ]);
    assert.notEqual(node.inputs[3].link, null);
    assert.equal(node.inputs[4].link, null);
});


test("moves later image links into a deleted middle gap", () => {
    const config = NODE_CONFIGS.PonyImageInputBundle;
    const node = fakeNode(
        allInputNames(config, 6),
        [1, 2, 4, 5],
    );
    const originalFourthLink = node.inputs[3].link;
    const originalFifthLink = node.inputs[4].link;
    const fourthLink = node.graph.links.get(originalFourthLink);
    const fifthLink = node.graph.links.get(originalFifthLink);

    compactImageInputs(node, config);

    assert.deepEqual(imageNames(node, config), [
        "image_1",
        "image_2",
        "image_3",
        "image_4",
        "image_5",
    ]);
    assert.equal(fourthLink.target_slot, 2);
    assert.equal(fifthLink.target_slot, 3);
});


test("uses the graph link store when input.link mirrors are empty", () => {
    const config = NODE_CONFIGS.PonyImageInputBundle;
    const node = fakeNode(
        allInputNames(config, 6),
        [],
    );
    const storedLinks = new Map(
        [0, 1, 2, 3].map(slot => {
            const link = {
                id: `stored-${slot + 1}`,
                target_id: node.id,
                target_slot: slot,
            };
            node.graph.links.set(link.id, link);
            return [slot, link];
        }),
    );
    node.getInputLink = slot => storedLinks.get(slot) ?? null;

    compactImageInputs(node, config);

    assert.deepEqual(imageNames(node, config), [
        "image_1",
        "image_2",
        "image_3",
        "image_4",
        "image_5",
    ]);
    for (let slot = 0; slot < 4; slot += 1) {
        assert.equal(storedLinks.get(slot).target_slot, slot);
    }
});


test("bundle node uses one stable image naming scheme", () => {
    assert.equal(
        NODE_CONFIGS.PonyImageInputBundle.format(3),
        "image_3",
    );
    assert.equal(
        NODE_CONFIGS.PonyImageInputBundle.maxInputs,
        10,
    );
});


test("normalizes interleaved bundle inputs into a stable image block", () => {
    const config = NODE_CONFIGS.PonyImageInputBundle;
    const node = fakeNode([
        "image_1",
        "metadata",
        "note",
        "image_3",
        "image_2",
    ], [1, 2, 3, 4, 5]);
    const fixedLinks = new Map(
        node.inputs
            .filter(input => !config.pattern.test(input.name))
            .map(input => [input.name, input.link])
    );

    normalizeImageInputOrder(node, config);
    repairInputTargetSlots(node);

    assert.deepEqual(node.inputs.map(input => input.name), [
        "image_1",
        "image_2",
        "image_3",
        "metadata",
        "note",
    ]);
    for (const [name, linkId] of fixedLinks) {
        const index = node.inputs.findIndex(input => input.name === name);
        assert.equal(node.graph.links.get(linkId).target_slot, index);
    }
});

