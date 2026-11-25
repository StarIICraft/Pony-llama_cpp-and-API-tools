import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function text(key, ...args) {
    const zh = String(
        app?.ui?.settings?.getSettingValue?.("Comfy.Locale")
        || "en"
    ).toLowerCase().startsWith("zh");
    const values = zh
        ? {
            resolveError: (name) => `无法解析“${name}”连接到的上游节点，请暂时直接填写该控件后再刷新。`,
            chooseModel: (count) => `选择模型（${count}）`,
            filter: "输入关键词筛选模型",
            apply: "使用选中的模型",
            cancel: "取消",
            baseUrlRequired: "请先填写 Base URL。",
            refreshing: "正在刷新...",
            invalidJson: "模型列表接口没有返回有效 JSON。",
            emptyModels: "没有获取到模型。",
            refreshFailed: (message) => `刷新模型失败：${message}`,
            refreshButton: "刷新模型列表",
        }
        : {
            resolveError: (name) => `Unable to resolve the upstream node connected to "${name}". Enter a value directly and refresh again.`,
            chooseModel: (count) => `Choose a model (${count})`,
            filter: "Filter models by keyword",
            apply: "Use selected model",
            cancel: "Cancel",
            baseUrlRequired: "Enter the Base URL first.",
            refreshing: "Refreshing...",
            invalidJson: "The model-list endpoint did not return valid JSON.",
            emptyModels: "No models were returned.",
            refreshFailed: (message) => `Failed to refresh models: ${message}`,
            refreshButton: "Refresh model list",
        };
    const value = values[key];
    return typeof value === "function" ? value(...args) : value;
}


function findWidget(node, name) {
    return node.widgets?.find(
        widget => widget.name === name
    );
}


function getGraphLink(graph, linkId) {
    const links = graph?.links;

    if (!links) {
        return null;
    }

    if (typeof links.get === "function") {
        return (
            links.get(linkId)
            || links.get(String(linkId))
            || links.get(Number(linkId))
            || null
        );
    }

    return (
        links[linkId]
        || links[String(linkId)]
        || null
    );
}


function getGraphNode(graph, nodeId) {
    if (typeof graph?.getNodeById === "function") {
        const node = graph.getNodeById(nodeId);

        if (node) {
            return node;
        }
    }

    const nodes = graph?._nodes || graph?.nodes || [];

    return nodes.find(
        node => String(node.id) === String(nodeId)
    ) || null;
}


function graphAncestors(graph) {
    const result = [];
    let current = graph;

    while (current && !result.includes(current)) {
        result.push(current);
        current = current.parentGraph || current.parent_graph;
    }

    return result;
}


function findEditableValueWidget(node) {
    if (!Array.isArray(node?.widgets)) {
        return null;
    }

    for (const name of [
        "text_0",
        "text",
        "value",
        "string",
        "Constant",
    ]) {
        const widget = node.widgets.find(
            item => item?.name === name
        );

        if (widget) {
            return widget;
        }
    }

    return null;
}


function findLinkedValueTarget(node, widgetName, visited = new Set()) {
    const input = (node?.inputs || []).find(
        item => (
            item.name === widgetName
            || item.widget?.name === widgetName
        )
    );
    const linkId = input?.link;

    if (linkId === undefined || linkId === null) {
        const widget = findWidget(node, widgetName);
        return widget ? { node, widget } : null;
    }

    const graph = node.graph || app.graph;
    const link = getGraphLink(graph, linkId);

    if (!link) {
        return null;
    }

    const visitKey = `${graph?.id ?? "root"}:${linkId}`;

    if (visited.has(visitKey)) {
        return null;
    }

    visited.add(visitKey);

    const origin = getGraphNode(graph, link.origin_id);

    if (!origin) {
        return null;
    }

    if (origin.type === "Reroute") {
        const rerouteLink = origin.inputs?.[0]?.link;

        if (rerouteLink !== undefined && rerouteLink !== null) {
            return findLinkedValueTarget(
                origin,
                origin.inputs[0].name || widgetName,
                visited
            );
        }
    }

    if (origin.type === "SetNode") {
        const sourceLink = origin.inputs?.[0]?.link;

        if (sourceLink !== undefined && sourceLink !== null) {
            return findLinkedValueTarget(
                origin,
                origin.inputs[0].name,
                visited
            );
        }
    }

    if (origin.type === "GetNode") {
        if (typeof origin.getSourceNode === "function") {
            const source = origin.getSourceNode();

            if (source?.node) {
                const widget = findEditableValueWidget(source.node);

                if (widget) {
                    return {
                        node: source.node,
                        widget,
                    };
                }
            }
        }

        const constantWidget = findWidget(origin, "Constant");
        const constant = constantWidget?.value;

        for (const ancestor of graphAncestors(graph)) {
            const setNode = (
                ancestor?._nodes
                || ancestor?.nodes
                || []
            ).find(candidate => (
                candidate?.type === "SetNode"
                && findWidget(candidate, "Constant")?.value
                    === constant
            ));

            if (!setNode) {
                continue;
            }

            const sourceLink = setNode.inputs?.[0]?.link;

            if (sourceLink !== undefined && sourceLink !== null) {
                return findLinkedValueTarget(
                    setNode,
                    setNode.inputs[0].name,
                    visited
                );
            }
        }
    }

    const valueWidget = findEditableValueWidget(origin);

    if (valueWidget) {
        return {
            node: origin,
            widget: valueWidget,
        };
    }

    return null;
}


function resolveInputValue(node, name, fallback = "") {
    const input = (node?.inputs || []).find(
        item => (
            item.name === name
            || item.widget?.name === name
        )
    );

    if (input?.link === undefined || input?.link === null) {
        return String(fallback ?? "");
    }

    const target = findLinkedValueTarget(node, name);

    if (!target) {
        throw new Error(text("resolveError", name));
    }

    return String(target.widget.value ?? "");
}


function chooseModel(node, models, warning = "") {
    return new Promise(resolve => {
        const overlay = document.createElement("div");

        overlay.style.cssText = `
            position: fixed;
            inset: 0;
            z-index: 100000;
            background: rgba(0, 0, 0, 0.55);
            display: flex;
            align-items: center;
            justify-content: center;
        `;

        const panel = document.createElement("div");

        panel.style.cssText = `
            background: #252525;
            color: white;
            padding: 20px;
            border-radius: 8px;
            width: min(520px, 90vw);
            max-height: 85vh;
            overflow: auto;
        `;

        const title = document.createElement("h3");
    title.textContent = text("chooseModel", models.length);

        const warningElement = document.createElement("div");
        warningElement.textContent = warning;
        warningElement.style.cssText = `
            display: ${warning ? "block" : "none"};
            margin-bottom: 10px;
            padding: 8px;
            border: 1px solid #8a6a2f;
            border-radius: 4px;
            background: #3b321f;
            color: #f2dfaf;
            white-space: pre-wrap;
            line-height: 1.45;
        `;

        const filter = document.createElement("input");
    filter.placeholder = text("filter");

        filter.style.cssText = `
            width: 100%;
            box-sizing: border-box;
            margin-bottom: 10px;
            padding: 8px;
            color: white;
            background: #333;
            border: 1px solid #666;
        `;

        const select = document.createElement("select");
        select.size = 14;

        select.style.cssText = `
            width: 100%;
            box-sizing: border-box;
            padding: 6px;
            color: white;
            background: #333;
            border: 1px solid #666;
        `;

        const modelTarget = findLinkedValueTarget(
            node,
            "模型名称"
        );
        const modelWidget = (
            modelTarget?.widget
            || findWidget(node, "模型名称")
        );
        const currentModel = modelWidget?.value;

        function renderOptions() {
            const keyword = filter.value.trim().toLowerCase();

            select.replaceChildren();

            for (const model of models) {
                if (!model.toLowerCase().includes(keyword)) {
                    continue;
                }

                const option = document.createElement("option");
                option.value = model;
                option.textContent = model;
                select.appendChild(option);
            }

            if (
                Array.from(select.options).some(
                    option => option.value === currentModel
                )
            ) {
                select.value = currentModel;
            } else if (select.options.length) {
                select.selectedIndex = 0;
            }
        }

        filter.addEventListener("input", renderOptions);

        const controls = document.createElement("div");

        controls.style.cssText = `
            display: flex;
            gap: 10px;
            margin-top: 12px;
        `;

        const apply = document.createElement("button");
    apply.textContent = text("apply");

        const cancel = document.createElement("button");
    cancel.textContent = text("cancel");

        function close() {
            document.removeEventListener("keydown", onKeyDown);
            overlay.remove();
            resolve();
        }

        function applySelection() {
            if (!select.value || !modelWidget) {
                return;
            }

            modelWidget.value = select.value;

            if (typeof modelWidget.callback === "function") {
                modelWidget.callback(
                    select.value,
                    app.canvas,
                    node
                );
            }

            node.setDirtyCanvas?.(true, true);
            close();
        }

        function onKeyDown(event) {
            if (event.key === "Escape") {
                close();
            }
        }

        apply.addEventListener("click", applySelection);
        cancel.addEventListener("click", close);
        select.addEventListener("dblclick", applySelection);

        overlay.addEventListener("click", event => {
            if (event.target === overlay) {
                close();
            }
        });

        document.addEventListener("keydown", onKeyDown);

        controls.append(apply, cancel);
        panel.append(
            title,
            warningElement,
            filter,
            select,
            controls
        );
        overlay.appendChild(panel);
        document.body.appendChild(overlay);

        renderOptions();
        filter.focus();
    });
}


app.registerExtension({
    name: "PonyToolbox.ApiModelRefresh",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PonyApiLlmLoader") {
            return;
        }

        const original = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = original?.apply(this, arguments);
            const node = this;

            let busy = false;

            const button = node.addWidget(
                "button",
                text("refreshButton"),
                null,
                async () => {
                    if (busy) {
                        return;
                    }

                    const baseWidget = findWidget(node, "接口地址");
                    const keyWidget = findWidget(node, "接口密钥");

                    // 这里只读取加载器自身的控件，
                    // 不全局扫描 DOM，避免拿到其他节点的密钥。
                    const baseUrl = resolveInputValue(
                        node,
                        "接口地址",
                        baseWidget?.value
                    ).trim();

                    const apiKey = resolveInputValue(
                        node,
                        "接口密钥",
                        keyWidget?.value
                    ).trim();

                    if (!baseUrl) {
                    alert(text("baseUrlRequired"));
                        return;
                    }

                    busy = true;
                    button.name = text("refreshing");
                    node.setDirtyCanvas?.(true, true);

                    try {
                        const response = await api.fetchApi(
                            "/mini_api_llm/models",
                            {
                                method: "POST",
                                headers: {
                                    "Content-Type": "application/json",
                                },
                                body: JSON.stringify({
                                    base_url: baseUrl,
                                    api_key: apiKey,
                                }),
                            }
                        );

                        let data;

                        try {
                            data = await response.json();
                        } catch {
                            throw new Error(
                                text("invalidJson")
                            );
                        }

                        if (!response.ok || data.error) {
                            throw new Error(
                                data.error
                                || `HTTP ${response.status}`
                            );
                        }

                        if (
                            !Array.isArray(data.models)
                            || !data.models.length
                        ) {
                            throw new Error(text("emptyModels"));
                        }

                        await chooseModel(
                            node,
                            data.models,
                            String(data.warning || "")
                        );

                    } catch (error) {
                        alert(text("refreshFailed", error.message));

                    } finally {
                        busy = false;
                        button.name = text("refreshButton");
                        node.setDirtyCanvas?.(true, true);
                    }
                },
                {
                    serialize: false,
                }
            );

            node.setSize?.(node.computeSize());

            return result;
        };
    },
});
