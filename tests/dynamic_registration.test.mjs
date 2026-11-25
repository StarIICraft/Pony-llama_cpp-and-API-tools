import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { fileURLToPath } from "node:url";
import path from "node:path";


const TEST_DIRECTORY = path.dirname(fileURLToPath(import.meta.url));
const PLUGIN_DIRECTORY = path.resolve(TEST_DIRECTORY, "..");
const WEB_DIRECTORY = path.join(PLUGIN_DIRECTORY, "web");
let moduleCounter = 0;


function dataModule(source) {
    moduleCounter += 1;
    return `data:text/javascript;base64,${
        Buffer.from(source).toString("base64")
    }#${moduleCounter}`;
}


async function loadExtension() {
    let extension = null;
    const app = {
        graph: {
            _nodes: [],
            links: new Map(),
            getNodeById() {
                return null;
            },
            setDirtyCanvas() {},
        },
        canvas: {
            onDrawForeground: null,
        },
        configuringGraph: false,
        ui: {
            settings: {
                getSettingValue() {
                    return "zh";
                },
            },
        },
        registerExtension(value) {
            extension = value;
        },
    };

    globalThis.app = app;
    globalThis.LiteGraph = {
        createNode() {
            return null;
        },
    };

    const coreSource = await readFile(
        path.join(WEB_DIRECTORY, "dynamic_image_inputs_core.js"),
        "utf8",
    );
    const coreUrl = dataModule(coreSource);
    let mainSource = await readFile(
        path.join(WEB_DIRECTORY, "dynamic_image_inputs.js"),
        "utf8",
    );
    mainSource = mainSource
        .replace(
            'import { app } from "../../scripts/app.js";\n\n',
            "const app = globalThis.app;\n",
        )
        .replace(
            'from "./dynamic_image_inputs_core.js";',
            `from ${JSON.stringify(coreUrl)};`,
        );

    await import(dataModule(mainSource));
    assert.ok(extension);
    return extension;
}


test("companion owner construction is safe before graph attachment", async () => {
    const extension = await loadExtension();

    for (const nodeName of [
        "PonyLlamaCppInstruct",
        "PonyApiLlmChain",
        "PonyApiImageEditor",
        "PonyMultiImageResize",
    ]) {
        class ConstructorSafeNode {
            constructor() {
                this.inputs = [{
                    name: "image_bundle",
                    type: "PONY_IMAGE_BUNDLE",
                    link: null,
                }];
                this.outputs = [];
                this.size = [320, 200];
                this.properties = {};
            }

            setSize(size) {
                this.size = size;
                this.onResize?.(size);
            }
        }

        await extension.beforeRegisterNodeDef(
            ConstructorSafeNode,
            { name: nodeName },
        );

        const node = new ConstructorSafeNode();
        node.setSize([360, 240]);

        assert.deepEqual(node.size, [360, 240], nodeName);
    }
});


test("resize controller creates and connects its I/O companion", async () => {
    const extension = await loadExtension();
    let nextLinkId = 1;
    const graph = {
        _nodes: [],
        links: new Map(),
        add(node) {
            this._nodes.push(node);
        },
        getNodeById(id) {
            return this._nodes.find(
                node => String(node.id) === String(id),
            ) || null;
        },
        setDirtyCanvas() {},
    };
    const companion = {
        id: 50,
        type: "PonyMultiImageResizeIO",
        inputs: [
            { name: "image_1", type: "IMAGE", link: null },
            { name: "image_2", type: "IMAGE", link: null },
            { name: "image_3", type: "IMAGE", link: null },
            { name: "mask_1", type: "MASK", link: null },
            { name: "resize_config", type: "PONY_MULTI_IMAGE_RESIZE_CONFIG", link: null },
        ],
        outputs: Array.from({ length: 10 }, (_, index) => ({
            name: `output_${index}`,
            type: index < 8 ? "IMAGE" : "MASK",
            links: [],
        })),
        properties: {},
        size: [320, 400],
        pos: [0, 0],
        graph,
        addInput(name, type) {
            const input = { name, type, link: null };
            this.inputs.push(input);
            return input;
        },
        removeInput(index) {
            this.inputs.splice(index, 1);
        },
        computeSize() {
            return [
                180,
                Math.max(this.inputs.length, this.outputs.length) * 20 + 60,
            ];
        },
        setSize(size) {
            this.size = size;
        },
        arrange() {},
        setDirtyCanvas() {},
        _setConcreteSlots() {},
        connect(originSlot, targetNode, targetSlot) {
            const link = {
                id: nextLinkId++,
                origin_id: this.id,
                origin_slot: originSlot,
                target_id: targetNode.id,
                target_slot: targetSlot,
                type: "PONY_IMAGE_BUNDLE",
            };
            graph.links.set(link.id, link);
            return link;
        },
    };
    globalThis.LiteGraph.createNode = type => (
        type === "PonyMultiImageResizeIO"
            ? companion
            : null
    );

    const main = {
        id: 10,
        type: "PonyMultiImageResize",
        comfyClass: "PonyMultiImageResize",
        inputs: [{
            name: "result_bundle",
            type: "PONY_IMAGE_BUNDLE",
            link: null,
        }, ...Array.from({ length: 14 }, (_, index) => ({
            name: `resize_widget_${index}`,
            type: "STRING",
            link: null,
        }))],
        outputs: Array.from({ length: 5 }, (_, index) => ({
            name: `result_${index}`,
            type: index ? "MASK" : "PONY_IMAGE_BUNDLE",
            links: [],
        })),
        properties: {},
        size: [320, 300],
        pos: [100, 500],
        graph,
        setDirtyCanvas() {},
    };
    graph._nodes.push(main);
    globalThis.app.graph = graph;
    extension.setup();

    extension.loadedGraphNode(main);
    await new Promise(resolve => setTimeout(resolve, 80));

    assert.ok(graph._nodes.includes(companion));
    const resultSlot = main.inputs.findIndex(
        input => input.name === "result_bundle",
    );
    assert.ok(resultSlot >= 0);
    assert.equal(companion.properties.imageBundleOwnerId, main.id);
    assert.equal(main.properties.imageBundleCompanionId, companion.id);
    assert.ok([...graph.links.values()].some(link => (
        link.origin_id === companion.id
        && link.origin_slot === 9
        && link.target_id === main.id
        && link.target_slot === resultSlot
    )));

    main.pos = [180, 620];
    globalThis.app.canvas.onDrawForeground();
    assert.equal(companion.pos[0], 180);
    assert.equal(
        companion.pos[1],
        620 - 30 - companion.size[1] - 2,
    );

    for (const mode of [0, 2, 4]) {
        main.mode = mode;
        globalThis.app.canvas.onDrawForeground();
        assert.equal(companion.mode, mode);
    }

    const group = {
        boundingRect: new Float32Array([150, 200, 600, 800]),
    };
    graph.groups = [group];
    main.pos = [180, 620];
    companion.size = [companion.size[0], 400];
    globalThis.app.canvas.onDrawForeground();
    let groupBounds = [...group.boundingRect];
    assert.ok(groupBounds[0] <= main.pos[0] - 8);
    assert.ok(groupBounds[1] <= companion.pos[1] - 30 - 8);
    assert.ok(
        groupBounds[0] + groupBounds[2]
        >= main.pos[0] + main.size[0] + 8,
    );
    assert.ok(
        groupBounds[1] + groupBounds[3]
        >= main.pos[1] + main.size[1] + 8,
    );

    const previousGroupTop = groupBounds[1];
    companion.size = [companion.size[0], 500];
    globalThis.app.canvas.onDrawForeground();
    groupBounds = [...group.boundingRect];
    assert.ok(groupBounds[1] < previousGroupTop);
    assert.ok(groupBounds[1] <= companion.pos[1] - 30 - 8);

    const callbacks = [];
    const previousRequestAnimationFrame =
        globalThis.requestAnimationFrame;
    const previousGraph = globalThis.app.graph;
    globalThis.requestAnimationFrame = callback => {
        callbacks.push(callback);
        return callbacks.length;
    };
    try {
        globalThis.app.graph = {
            _nodes: [main],
            links: new Map(),
            getNodeById(id) {
                return String(main.id) === String(id)
                    ? main
                    : null;
            },
            setDirtyCanvas() {},
        };
        main.computeSize = () => [main.size[0], 188];
        main.setSize = size => {
            main.size = size;
        };
        extension.afterConfigureGraph();
        for (const callback of callbacks.splice(0)) {
            callback();
        }
        assert.deepEqual(main.size, [320, 188]);
    } finally {
        globalThis.app.graph = previousGraph;
        if (previousRequestAnimationFrame === undefined) {
            delete globalThis.requestAnimationFrame;
        } else {
            globalThis.requestAnimationFrame =
                previousRequestAnimationFrame;
        }
    }
});


test("queueing injects controller widgets into the companion config", async () => {
    const extension = await loadExtension();
    const app = globalThis.app;
    const main = {
        id: 10,
        type: "PonyMultiImageResize",
        properties: {
            imageBundleCompanionId: 50,
        },
        widgets: [
            { name: "resize_mode", value: "统一百万像素限制" },
            { name: "megapixels", value: 2.5 },
            { name: "divisor", value: 8 },
        ],
        subgraph: null,
    };
    const companion = {
        id: 50,
        type: "PonyMultiImageResizeIO",
    };
    app.graph._nodes.push(main, companion);
    app.graph.getNodeById = id => (
        [main, companion].find(
            node => String(node.id) === String(id),
        ) || null
    );
    app.graphToPrompt = async () => ({
        output: {
            10: {
                inputs: {
                    result_bundle: ["50", 0],
                    megapixels: 3.04,
                    divisor: ["1042", 0],
                },
            },
            50: {
                inputs: {
                    resize_config: "{}",
                },
            },
        },
    });

    extension.setup();
    const prompt = await app.graphToPrompt();

    assert.deepEqual(
        JSON.parse(prompt.output[50].inputs.resize_config),
        {
            language: "中文",
            resize_mode: "统一百万像素限制",
            megapixels: 3.04,
            divisor: 8,
        },
    );
    assert.deepEqual(
        prompt.output[10].inputs.result_bundle,
        [50, 9],
    );
    assert.equal(prompt.output[50].inputs.megapixels, undefined);
    assert.deepEqual(
        prompt.output[50].inputs.divisor,
        ["1042", 0],
    );
});


test("queue wrapper preserves other nodes' external links", async () => {
    const extension = await loadExtension();
    const app = globalThis.app;
    const nodes = [
        {
            id: 11,
            type: "PonyLlamaCppInstruct",
            comfyClass: "PonyLlamaCppInstruct",
        },
        {
            id: 12,
            type: "PonyApiLlmChain",
            comfyClass: "PonyApiLlmChain",
        },
        {
            id: 13,
            type: "PonyApiImageEditor",
            comfyClass: "PonyApiImageEditor",
        },
    ];
    app.graph._nodes = nodes;
    app.graph.getNodeById = id => (
        nodes.find(
            node => String(node.id) === String(id),
        ) || null
    );
    app.graphToPrompt = async () => ({
        output: {
            11: {
                inputs: {
                    max_megapixels: ["1041", 0],
                },
            },
            12: {
                inputs: {
                    用户提示词: ["1042", 0],
                },
            },
            13: {
                inputs: {
                    接口地址: ["1043", 0],
                },
            },
        },
    });

    extension.setup();
    const prompt = await app.graphToPrompt();

    assert.deepEqual(
        prompt.output[11].inputs.max_megapixels,
        ["1041", 0],
    );
    assert.deepEqual(
        prompt.output[12].inputs.用户提示词,
        ["1042", 0],
    );
    assert.deepEqual(
        prompt.output[13].inputs.接口地址,
        ["1043", 0],
    );
});


test("minimizes the height of every dynamic main node type", async () => {
    const extension = await loadExtension();
    const app = globalThis.app;
    const types = [
        "PonyLlamaCppInstruct",
        "PonyApiLlmChain",
        "PonyApiImageEditor",
        "PonyMultiImageResize",
    ];
    const nodes = types.map((type, index) => ({
        id: index + 1,
        type,
        comfyClass: type,
        size: [320, 600],
        pos: [index * 400, 100],
        flags: index === 0 ? { collapsed: true } : {},
        properties: {},
        setSize(size) {
            this.size = size;
        },
        setDirtyCanvas() {},
        computeSize() {
            return [this.size[0], 180 + this.id];
        },
    }));
    app.graph = {
        _nodes: nodes,
        links: new Map(),
        getNodeById(id) {
            return nodes.find(
                node => String(node.id) === String(id),
            ) || null;
        },
        setDirtyCanvas() {},
    };

    const previousRequestAnimationFrame =
        globalThis.requestAnimationFrame;
    const callbacks = [];
    globalThis.requestAnimationFrame = callback => {
        callbacks.push(callback);
        return callbacks.length;
    };
    try {
        extension.afterConfigureGraph();
        for (const callback of callbacks.splice(0)) {
            callback();
        }
    } finally {
        if (previousRequestAnimationFrame === undefined) {
            delete globalThis.requestAnimationFrame;
        } else {
            globalThis.requestAnimationFrame =
                previousRequestAnimationFrame;
        }
    }

    assert.deepEqual(
        nodes.map(node => node.size[1]),
        [181, 182, 183, 184],
    );
});
