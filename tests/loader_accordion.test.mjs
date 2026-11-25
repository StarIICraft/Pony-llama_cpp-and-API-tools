import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import path from "node:path";
import { pathToFileURL } from "node:url";


const pluginRoot = path.resolve(
    path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1")),
    "..",
);


function widget(name, value) {
    return {
        name,
        label: name,
        value,
        hidden: false,
        computeSize: () => [0, 24],
        draw() {},
    };
}


function makeNode(overrides = {}) {
    const widgets = [
        widget("model", "gemma.gguf"),
        widget("mmproj", "mmproj.gguf"),
        widget("chat_handler", "Gemma4"),
        widget("n_ctx", 8192),
        widget("vram_limit", -1),
        widget("image_min_tokens", 0),
        widget("image_max_tokens", 0),
        widget("load_mtp", false),
        widget("kv_cache_mode", "K F16 / V F16"),
        widget("cpu_moe", false),
        widget("n_cpu_moe", 0),
    ];
    const node = {
        id: 1,
        type: "PonyLlamaCppModelLoader",
        comfyClass: "PonyLlamaCppModelLoader",
        properties: {},
        inputs: [
            "model",
            "mmproj",
            "chat_handler",
            "n_ctx",
            "vram_limit",
            "image_min_tokens",
            "image_max_tokens",
            "load_mtp",
            "kv_cache_mode",
            "cpu_moe",
            "n_cpu_moe",
        ].map(name => ({
            name,
            widget: { name },
            link: null,
            hidden: false,
        })),
        widgets,
        size: [420, 500],
        addWidget(type, name, value, callback, options = {}) {
            const created = widget(name, value);
            created.type = type;
            created.callback = callback;
            created.options = options;
            this.widgets.push(created);
            return created;
        },
        computeSize() {
            let height = 80;
            for (const entry of this.widgets) {
                const size = entry.computeSize?.(this.size[0]) || [0, 24];
                height += Math.max(0, Number(size[1]) || 0);
            }
            return [this.size[0], height];
        },
        setSize(size) {
            this.size = [...size];
        },
        setDirtyCanvas() {},
        ...overrides,
    };
    return node;
}


async function loadModule() {
    globalThis.app = {
        ui: {
            settings: {
                getSettingValue: () => "zh-CN",
            },
        },
        graph: {
            setDirtyCanvas() {},
        },
        registerExtension(extension) {
            globalThis.registeredExtension = extension;
        },
    };

    const sourcePath = path.join(pluginRoot, "web", "llama_loader_accordion.js");
    const source = fs.readFileSync(sourcePath, "utf8").replace(
        'import { app } from "../../scripts/app.js";',
        "const app = globalThis.app;",
    );
    const moduleUrl = `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
    return import(moduleUrl);
}


test("adds two non-serialized accordion toggles and collapses defaults", async () => {
    const module = await loadModule();
    const node = makeNode();
    const serializableBefore = node.widgets.length;

    module.syncLoaderAccordion(node);

    const toggles = node.widgets.filter(entry => entry.__ponyAccordionGroup);
    assert.equal(toggles.length, 2);
    assert.ok(toggles.every(entry => entry.serialize === false));
    assert.ok(toggles.every(entry => entry.options.serialize === false));
    assert.equal(
        node.widgets.filter(entry => entry.serialize !== false).length,
        serializableBefore,
    );

    const minWidget = node.widgets.find(entry => entry.name === "image_min_tokens");
    const cpuWidget = node.widgets.find(entry => entry.name === "cpu_moe");
    assert.equal(minWidget.hidden, true);
    assert.equal(cpuWidget.hidden, true);
    assert.ok(
        node.widgets.indexOf(toggles[0])
        < node.widgets.indexOf(minWidget),
    );
    assert.ok(
        node.widgets.indexOf(toggles[1])
        < node.widgets.indexOf(cpuWidget),
    );
});


test("clicking toggle expands children and preserves node width", async () => {
    const module = await loadModule();
    const node = makeNode();
    module.syncLoaderAccordion(node);
    const widthBefore = node.size[0];
    const heightBefore = node.size[1];

    const imageToggle = node.widgets.find(
        entry => entry.__ponyAccordionGroup === "image_tokens"
    );
    imageToggle.callback();

    const minWidget = node.widgets.find(entry => entry.name === "image_min_tokens");
    const maxWidget = node.widgets.find(entry => entry.name === "image_max_tokens");
    assert.equal(minWidget.hidden, false);
    assert.equal(maxWidget.hidden, false);
    assert.equal(node.size[0], widthBefore);
    assert.ok(node.size[1] > heightBefore);
});


test("active values and external links default to expanded", async () => {
    const module = await loadModule();

    const activeNode = makeNode();
    activeNode.widgets.find(
        entry => entry.name === "image_max_tokens"
    ).value = 1024;
    module.syncLoaderAccordion(activeNode);
    assert.equal(
        activeNode.widgets.find(
            entry => entry.name === "image_max_tokens"
        ).hidden,
        false,
    );

    const linkedNode = makeNode();
    linkedNode.inputs.find(
        entry => entry.name === "cpu_moe"
    ).link = 123;
    module.syncLoaderAccordion(linkedNode);
    assert.equal(
        linkedNode.widgets.find(
            entry => entry.name === "cpu_moe"
        ).hidden,
        false,
    );
});


test("accordion state is stored in node properties, not widget values", async () => {
    const module = await loadModule();
    const node = makeNode();
    module.syncLoaderAccordion(node);

    const moeToggle = node.widgets.find(
        entry => entry.__ponyAccordionGroup === "moe_offload"
    );
    moeToggle.callback();

    assert.deepEqual(
        node.properties.ponyLoaderAccordion,
        {
            image_tokens: false,
            moe_offload: true,
        },
    );
});
