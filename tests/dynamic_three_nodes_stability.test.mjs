import assert from "node:assert/strict";
import test from "node:test";

import {
    NODE_CONFIGS,
    compactImageInputs,
    repairInputTargetSlots,
} from "../web/dynamic_image_inputs_core.js";


const CASES = [
    {
        config: NODE_CONFIGS.PonyImageInputBundle,
        imageNames: [
            "image_1", "image_2", "image_3", "image_4",
            "image_5", "image_6", "image_7", "image_8",
            "image_9", "image_10",
        ],
        tails: ["metadata", "note"],
    },
    {
        config: NODE_CONFIGS.PonyImageInputBundle,
        imageNames: [
            "image_1", "image_2", "image_3", "image_4",
            "image_5", "image_6", "image_7", "image_8",
            "image_9", "image_10",
        ],
        tails: ["metadata", "note"],
    },
    {
        config: NODE_CONFIGS.PonyImageInputBundle,
        imageNames: [
            "image_1", "image_2", "image_3", "image_4",
            "image_5", "image_6", "image_7", "image_8",
            "image_9", "image_10",
        ],
        tails: ["metadata", "note"],
    },
];


function createNode(testCase) {
    const links = new Map();
    const inputs = [];

    for (const [index, name] of testCase.imageNames.entries()) {
        const connected = index < 3;
        const linkId = connected ? `image-${index + 1}` : null;
        inputs.push({ name, type: "IMAGE", link: linkId });
        if (linkId) {
            links.set(linkId, {
                id: linkId,
                target_id: 10,
                target_slot: index,
            });
        }
    }

    for (const [offset, name] of testCase.tails.entries()) {
        const linkId = `tail-${offset}`;
        const index = inputs.length;
        inputs.push({ name, type: "IMAGE", link: linkId });
        links.set(linkId, {
            id: linkId,
            target_id: 10,
            target_slot: index,
        });
    }

    const outputs = [
        { name: "输出", type: "STRING", links: [] },
        { name: "状态", type: "INT", links: [] },
    ];

    const node = {
        id: 10,
        inputs,
        outputs,
        graph: { links },
        size: [400, 700],
        addInput(name, type) {
            const input = { name, type, link: null };
            this.inputs.push(input);
            return input;
        },
        removeInput(index) {
            this.inputs.splice(index, 1);
        },
        computeSize() {
            return [400, this.inputs.length * 20 + 120];
        },
        setSize(size) {
            this.size = size;
        },
        arrange() {},
        setDirtyCanvas() {},
        _setConcreteSlots() {},
    };

    return node;
}


function assertTailLinks(node, testCase) {
    for (const [offset, name] of testCase.tails.entries()) {
        const inputIndex = node.inputs.findIndex(
            input => input.name === name
        );
        assert.notEqual(inputIndex, -1, `${name} disappeared`);
        assert.equal(
            node.graph.links.get(`tail-${offset}`).target_slot,
            inputIndex,
            `${name} link target shifted`,
        );
    }
}


for (const [caseIndex, testCase] of CASES.entries()) {
    test(`dynamic node ${caseIndex + 1} is stable while images grow and shrink`, () => {
        const node = createNode(testCase);
        const originalOutputs = [...node.outputs];

        compactImageInputs(node, testCase.config);
        repairInputTargetSlots(node);
        assertTailLinks(node, testCase);
        assert.deepEqual(node.outputs, originalOutputs);

        const fourth = node.inputs.find(
            input => testCase.config.pattern.test(input.name)
                && testCase.config.format(4) === input.name
        );
        if (fourth) {
            node.graph.links.set("image-4", {
                id: "image-4",
                target_id: 10,
                target_slot: node.inputs.indexOf(fourth),
            });
            fourth.link = "image-4";
        }

        compactImageInputs(node, testCase.config);
        repairInputTargetSlots(node);
        assertTailLinks(node, testCase);
        assert.deepEqual(node.outputs, originalOutputs);

        if (fourth) {
            fourth.link = null;
            node.graph.links.delete("image-4");
        }

        compactImageInputs(node, testCase.config);
        repairInputTargetSlots(node);
        assertTailLinks(node, testCase);
        assert.deepEqual(node.outputs, originalOutputs);
    });
}
