import assert from "node:assert/strict";
import test from "node:test";

import {
    repairInputTargetSlots,
    repairOutputOriginSlots,
} from "../web/dynamic_image_inputs_core.js";


function fakeNode(inputs, outputs, links) {
    return {
        id: 10,
        inputs,
        outputs,
        graph: { links: new Map(links.map(link => [link.id, link])) },
    };
}


test("repairs input target slots from input link ownership", () => {
    const node = fakeNode(
        [
            { name: "images", link: "a" },
            { name: "images_02", link: "b" },
            { name: "queue_handler", link: "c" },
            { name: "skill", link: "d" },
        ],
        [],
        [
            { id: "a", target_id: 10, target_slot: 0 },
            { id: "b", target_id: 10, target_slot: 1 },
            { id: "c", target_id: 10, target_slot: 88 },
            { id: "d", target_id: 10, target_slot: 99 },
        ],
    );

    repairInputTargetSlots(node);

    assert.equal(node.graph.links.get("c").target_slot, 2);
    assert.equal(node.graph.links.get("d").target_slot, 3);
});


test("repairs output origin slots from output link ownership", () => {
    const node = fakeNode(
        [],
        [
            { name: "输出图像1", links: ["a"] },
            { name: "图1遮罩", links: ["b"] },
        ],
        [
            { id: "a", origin_id: 10, origin_slot: 77 },
            { id: "b", origin_id: 10, origin_slot: 88 },
        ],
    );

    repairOutputOriginSlots(node);

    assert.equal(node.graph.links.get("a").origin_slot, 0);
    assert.equal(node.graph.links.get("b").origin_slot, 1);
});


test("repairs slots when link ids differ between number and string", () => {
    const node = fakeNode(
        [
            { name: "images", link: 42 },
            { name: "custom_prompt", link: 43 },
        ],
        [
            { name: "output", links: [44] },
        ],
        [
            { id: "42", target_id: 10, target_slot: 90 },
            { id: "43", target_id: 10, target_slot: 91 },
            { id: "44", origin_id: 10, origin_slot: 92 },
        ],
    );

    repairInputTargetSlots(node);
    repairOutputOriginSlots(node);

    assert.equal(node.graph.links.get("42").target_slot, 0);
    assert.equal(node.graph.links.get("43").target_slot, 1);
    assert.equal(node.graph.links.get("44").origin_slot, 0);
});


