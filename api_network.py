import asyncio
import json
import re
from contextlib import contextmanager
from urllib.parse import urlsplit, urlunsplit

import requests
from aiohttp import web
from server import PromptServer

import comfy.model_management as mm


ENDPOINT_SUFFIXES = (
    "/chat/completions",
    "/models",
)

RETRYABLE_ENDPOINT_STATUSES = {
    404,
    405,
}

DIRECT_CONNECTION_PROXIES = {
    "http": None,
    "https": None,
}

OFFICIAL_MODEL_FALLBACKS = {
    "api.deepseek.com": [
        "deepseek-chat",
        "deepseek-reasoner",
        "deepseek-flash",
        "deepseek-v4.1-flash-1017",
    ],
}


class StreamingResponseError(RuntimeError):
    pass


class RetryWithoutProxyError(requests.RequestException):
    pass


def sanitize_api_error(value, limit=1200):
    text = str(value or "").strip()

    if not text:
        return ""

    text = re.sub(
        r"\bsk-[A-Za-z0-9_-]{8,}\b",
        "<API_KEY>",
        text,
    )
    text = re.sub(
        r"data:image/[^;\\s]+;base64,[A-Za-z0-9+/=\\s]+",
        "<IMAGE_DATA>",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > limit:
        text = text[:limit].rstrip() + "..."

    return text


def extract_api_error(value):
    if value is None:
        return ""

    if isinstance(value, str):
        return sanitize_api_error(value)

    if not isinstance(value, dict):
        return sanitize_api_error(value)

    error = value.get("error", value)

    if isinstance(error, str):
        return sanitize_api_error(error)

    if isinstance(error, dict):
        parts = []

        for key in (
            "message",
            "detail",
            "code",
            "type",
            "param",
        ):
            item = error.get(key)

            if item in (None, ""):
                continue

            rendered = sanitize_api_error(item, 600)

            if rendered:
                parts.append(f"{key}={rendered}")

        if parts:
            return sanitize_api_error(" | ".join(parts))

        return sanitize_api_error(error)

    for key in ("message", "detail", "msg"):
        item = value.get(key)

        if item:
            return sanitize_api_error(item)

    return sanitize_api_error(value)


def response_error_detail(response):
    try:
        return extract_api_error(response.json())
    except (TypeError, ValueError):
        return sanitize_api_error(
            getattr(response, "text", "")
        )


def api_error_hint(detail, status_code=None):
    text = str(detail or "").casefold()

    if any(
        marker in text
        for marker in (
            "country, region, or territory not supported",
            "unsupported_country_region_territory",
            "unsupported region",
        )
    ):
        return (
            "当前网络出口地区不受该 API 支持，"
            "请切换支持地区的代理"
        )

    if status_code == 402 or any(
        marker in text
        for marker in (
            "insufficient balance",
            "insufficient_balance",
            "余额不足",
        )
    ):
        return "账户余额不足，请前往 API 服务商充值"

    if status_code in (401, 403) or any(
        marker in text
        for marker in (
            "invalid api key",
            "incorrect api key",
            "authentication",
            "unauthorized",
            "permission denied",
        )
    ):
        return "API 密钥无效、已过期或没有该模型权限"

    if status_code == 429 or any(
        marker in text
        for marker in (
            "rate limit",
            "too many requests",
            "quota exceeded",
        )
    ):
        return "请求频率或账户配额已超限"

    if any(
        marker in text
        for marker in (
            "context length",
            "maximum context",
            "too many tokens",
            "token limit",
        )
    ):
        return "上下文长度超过模型上限"

    if any(
        marker in text
        for marker in (
            "model not found",
            "model does not exist",
            "unknown model",
        )
    ):
        return "模型名称不存在或当前账户不可用"

    if any(
        marker in text
        for marker in (
            "image input",
            "image_url",
            "vision",
            "unsupported image",
        )
    ):
        return "当前模型或接口不支持所发送的图片输入"

    if status_code == 404:
        return "接口路径不存在"

    if status_code == 400:
        return "请求参数不被服务端接受"

    if status_code is not None and status_code >= 500:
        return "API 服务端发生内部错误"

    return ""


def region_restriction_detail(value):
    text = str(value or "").casefold()
    return any(
        marker in text
        for marker in (
            "country, region, or territory not supported",
            "unsupported_country_region_territory",
            "unsupported region",
        )
    )


def build_api_error_message(prefix, status_code, detail):
    hint = api_error_hint(detail, status_code)

    if hint:
        message = (
            f"{prefix}：{hint}"
            f"（HTTP {status_code}）。"
        )
    else:
        message = (
            f"{prefix}，HTTP {status_code}。"
            "请检查接口地址、密钥和请求参数。"
        )

    if detail:
        message += f"\n服务端返回：{detail}"

    return message


def validate_base_url(value):
    value = str(value or "").strip()

    if not value:
        raise ValueError("Base URL 不能为空。")

    if any(character.isspace() for character in value):
        raise ValueError("Base URL 不能包含空白字符。")

    parsed = urlsplit(value)

    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Base URL 必须是有效的 HTTP/HTTPS 基础地址，"
            "不能包含用户名、密码、查询参数或 fragment。"
        )

    try:
        parsed.port
    except ValueError:
        raise ValueError("Base URL 的端口号无效。") from None

    path = parsed.path.rstrip("/")
    lower_path = path.casefold()

    # 即使误填了完整端点，也自动还原为基础地址。
    for suffix in ENDPOINT_SUFFIXES:
        if lower_path.endswith(suffix):
            path = path[:-len(suffix)].rstrip("/")
            break

    return urlunsplit((
        parsed.scheme,
        parsed.netloc,
        path,
        "",
        "",
    ))


def api_endpoint_candidates(base_url, endpoint):
    """
    同时兼容带 /v1 和不带 /v1 的基础地址。

    优先尝试显式的 /v1 路径，收到 404/405 时再回退到
    相同的非版本化路径；反过来也适用。
    """
    base_url = validate_base_url(base_url)
    endpoint = "/" + str(endpoint or "").lstrip("/")

    parsed = urlsplit(base_url)
    path = parsed.path.rstrip("/")

    if path.casefold().endswith("/v1"):
        versioned_path = path
        plain_path = path[:-3].rstrip("/")
    else:
        versioned_path = f"{path}/v1".rstrip("/")
        plain_path = path

    result = []

    for candidate_path in (
        versioned_path,
        plain_path,
    ):
        candidate_base = urlunsplit((
            parsed.scheme,
            parsed.netloc,
            candidate_path,
            "",
            "",
        ))

        candidate_url = candidate_base + endpoint

        if candidate_url not in result:
            result.append(candidate_url)

    return result


@contextmanager
def request_api_endpoint(
    method,
    base_url,
    endpoint,
    direct=False,
    **kwargs,
):
    """
    请求 API 端点，并在常见“路由不存在”错误时自动尝试
    /v1 与非 /v1 两种形式。
    """
    candidates = api_endpoint_candidates(
        base_url,
        endpoint,
    )

    request_options = dict(kwargs)

    if direct:
        request_options["proxies"] = dict(
            DIRECT_CONNECTION_PROXIES
        )

    for index, url in enumerate(candidates):
        response = requests.request(
            method,
            url,
            **request_options,
        )

        if (
            not direct
            and response.status_code == 407
        ):
            response.close()
            raise requests.exceptions.ProxyError(
                "代理服务器要求认证，HTTP 407。",
                response=response,
            )

        if (
            not direct
            and response.status_code == 403
        ):
            detail = response_error_detail(response)

            if region_restriction_detail(detail):
                response.close()
                raise RetryWithoutProxyError(
                    "代理出口地区受限，尝试无代理直连。",
                    response=response,
                )

        if (
            response.status_code
            in RETRYABLE_ENDPOINT_STATUSES
            and index < len(candidates) - 1
        ):
            response.close()
            continue

        try:
            yield response
        finally:
            response.close()

        return


def response_text(value):
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return "\n".join(
            item["text"]
            for item in value
            if isinstance(item, dict)
            and isinstance(item.get("text"), str)
        )

    return str(value)


def parse_json_response(
    response,
    service_name,
    allow_list=False,
):
    if not response.ok:
        detail = response_error_detail(response)
        raise RuntimeError(
            build_api_error_message(
                f"{service_name}请求失败",
                response.status_code,
                detail,
            )
        )

    try:
        data = response.json()
    except ValueError:
        raise RuntimeError(
            f"{service_name} 没有返回有效 JSON。"
        ) from None

    if not isinstance(data, dict) and not (
        allow_list
        and isinstance(data, list)
    ):
        raise RuntimeError(
            f"{service_name} 返回格式无效。"
        )

    if isinstance(data, dict) and data.get("error"):
        detail = extract_api_error(data)
        message = (
            f"{service_name} 返回了错误，"
            "请检查服务端日志或配置。"
        )

        if detail:
            message += f"\n服务端返回：{detail}"

        raise RuntimeError(message)

    return data


def call_chat_api(
    config,
    messages,
    parameters,
    stream=False,
    allow_stream_fallback=True,
    direct=False,
    allow_direct_fallback=True,
):
    """
    调用 OpenAI 兼容 Chat Completions。

    stream=True:
        接收 SSE，实时打印到后端控制台，
        最后返回完整字符串。
    """
    base_url = validate_base_url(config["base_url"])

    headers = {
        "Content-Type": "application/json",
    }

    if config.get("api_key"):
        headers["Authorization"] = (
            f"Bearer {config['api_key']}"
        )

    payload = dict(parameters)
    payload.update({
        "model": config["model_name"],
        "messages": messages,
        "stream": bool(stream),
    })

    try:
        with request_api_endpoint(
            "POST",
            base_url,
            "/chat/completions",
            direct=direct,
            headers=headers,
            json=payload,
            stream=bool(stream),
            timeout=(15, int(config.get("timeout", 180))),
        ) as response:
            if not response.ok:
                detail = response_error_detail(response)
                message = build_api_error_message(
                    "大模型请求失败",
                    response.status_code,
                    detail,
                )
                message += "\n已自动尝试带 /v1 和不带 /v1 的端点。"

                raise RuntimeError(message)

            if not stream:
                data = parse_json_response(response, "大模型")
                choices = data.get("choices")

                if not isinstance(choices, list) or not choices:
                    raise RuntimeError("大模型返回结果缺少 choices。")

                message = choices[0].get("message")

                if not isinstance(message, dict):
                    raise RuntimeError("大模型返回结果缺少 message。")

                text = response_text(message.get("content"))
                reasoning = response_text(
                    message.get("reasoning_content")
                    or message.get("reasoning")
                )

                if not text.strip():
                    if reasoning.strip():
                        raise RuntimeError(
                            "大模型只返回了推理内容，"
                            "没有返回最终正文。"
                            "请增大最大生成长度或关闭推理输出后重试。"
                        )

                    raise RuntimeError(
                        "大模型请求成功，但返回了空内容。"
                    )

                return text, reasoning

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            # 某些兼容服务忽略 stream，直接返回普通 JSON。
            if "application/json" in content_type:
                data = parse_json_response(response, "大模型")
                choices = data.get("choices") or []

                if not choices:
                    raise RuntimeError("大模型返回结果缺少 choices。")

                message = choices[0].get("message") or {}

                text = response_text(message.get("content"))
                reasoning = response_text(
                    message.get("reasoning_content")
                    or message.get("reasoning")
                )

                if text:
                    print(text, flush=True)

                return text, reasoning

            text_parts = []
            reasoning_parts = []
            received_choice = False
            finished = False

            print("[API LLM] 流式响应开始：", flush=True)

            # 按 SSE 事件解析，支持同一事件包含多行 data。
            event_lines = []

            def consume_event(lines):
                nonlocal received_choice, finished

                if not lines:
                    return

                raw_data = "\n".join(lines).strip()

                if not raw_data:
                    return

                if raw_data == "[DONE]":
                    finished = True
                    return

                try:
                    event = json.loads(raw_data)
                except ValueError:
                    raise RuntimeError(
                        "流式响应包含无效 JSON。"
                    ) from None

                if not isinstance(event, dict):
                    raise RuntimeError("流式响应格式无效。")

                event_type = str(
                    event.get("type") or ""
                ).casefold()

                if (
                    event.get("error")
                    or event_type == "error"
                ):
                    detail = extract_api_error(event)
                    message = "大模型在流式生成期间返回错误。"

                    if detail:
                        message += f"\n服务端返回：{detail}"

                    raise StreamingResponseError(message)

                choices = event.get("choices") or []

                # 允许只包含 usage 的事件。
                if not choices:
                    return

                choice = next(
                    (
                        item
                        for item in choices
                        if item.get("index", 0) == 0
                    ),
                    None,
                )

                if choice is None:
                    return

                received_choice = True
                delta = choice.get("delta") or {}

                text = response_text(delta.get("content"))
                reasoning = response_text(
                    delta.get("reasoning_content")
                    or delta.get("reasoning")
                )

                if reasoning:
                    reasoning_parts.append(reasoning)

                if text:
                    text_parts.append(text)
                    print(text, end="", flush=True)

                if choice.get("finish_reason") is not None:
                    finished = True

            # 显式使用 UTF-8，避免中文流式片段乱码。
            response.encoding = "utf-8"

            for line in response.iter_lines(
                chunk_size=1,
                decode_unicode=True,
            ):
                mm.throw_exception_if_processing_interrupted()

                if line == "":
                    consume_event(event_lines)
                    event_lines = []
                    continue

                if line.startswith("data:"):
                    value = line[5:]

                    if value.startswith(" "):
                        value = value[1:]

                    event_lines.append(value)

                # SSE 注释、event、id、retry 等字段不参与 JSON 解析。

            if event_lines:
                consume_event(event_lines)

            print("", flush=True)

            if not received_choice:
                raise StreamingResponseError(
                    "没有收到有效的流式 choices 数据，"
                    "请确认服务支持 OpenAI SSE。"
                )

            if not finished:
                raise StreamingResponseError(
                    "流式连接提前结束，没有收到结束标记。"
                    "为避免把不完整内容当作完整结果，本次请求失败。"
                )

            if not text_parts:
                if reasoning_parts:
                    raise StreamingResponseError(
                        "流式生成只返回了推理内容，"
                        "没有返回最终正文。"
                        "请增大最大生成长度或关闭推理输出后重试。"
                    )

                raise StreamingResponseError(
                    "流式请求正常结束，"
                    "但模型没有返回任何正文或推理内容。"
                )

            return (
                "".join(text_parts),
                "".join(reasoning_parts),
            )

    except StreamingResponseError as error:
        if (
            allow_stream_fallback
            and stream
            and not text_parts
        ):
            print(
                "[API LLM] 流式请求未产出内容，"
                "自动改用非流式重试一次。",
                flush=True,
            )

            return call_chat_api(
                config=config,
                messages=messages,
                parameters=parameters,
                stream=False,
                allow_stream_fallback=False,
                direct=direct,
                allow_direct_fallback=allow_direct_fallback,
            )

        raise

    except requests.Timeout as error:
        partial_output = bool(
            locals().get("text_parts")
            or locals().get("reasoning_parts")
        )

        if (
            allow_direct_fallback
            and not direct
            and not partial_output
        ):
            print(
                "[API LLM] 代理网络请求超时，"
                "禁用环境代理后直连重试一次。",
                flush=True,
            )

            return call_chat_api(
                config=config,
                messages=messages,
                parameters=parameters,
                stream=False,
                allow_stream_fallback=False,
                direct=True,
                allow_direct_fallback=False,
            )

        detail = sanitize_api_error(
            f"{type(error).__name__}: {error}"
        )
        message = "大模型请求超时，请检查网络或增大 timeout。"

        if detail:
            message += f"\n网络错误：{detail}"

        raise RuntimeError(message) from None

    except requests.RequestException as error:
        partial_output = bool(
            locals().get("text_parts")
            or locals().get("reasoning_parts")
        )

        if (
            allow_direct_fallback
            and not direct
            and not partial_output
        ):
            print(
                "[API LLM] 代理或网络连接异常，"
                "禁用环境代理后直连重试一次。",
                flush=True,
            )

            return call_chat_api(
                config=config,
                messages=messages,
                parameters=parameters,
                stream=False,
                allow_stream_fallback=False,
                direct=True,
                allow_direct_fallback=False,
            )

        response = getattr(error, "response", None)
        detail = ""

        if response is not None:
            detail = response_error_detail(response)

        if not detail:
            detail = sanitize_api_error(
                f"{type(error).__name__}: {error}"
            )

        message = (
            "大模型直连请求失败或流式传输中断。"
            if direct
            else "大模型连接失败或流式传输中断。"
        )

        if detail:
            message += f"\n网络错误：{detail}"

        raise RuntimeError(message) from None


def search_tavily(query, api_key, max_results=5):
    """
    检索网页，返回提供给模型的参考文本。
    不执行网页中的指令，不抓取或运行网页代码。
    """
    query = str(query or "").strip()
    api_key = str(api_key or "").strip()

    if not query:
        raise ValueError("搜索关键词不能为空。")

    if not api_key:
        raise ValueError(
            "启用联网搜索后，需要填写 Tavily API Key。"
        )

    mm.throw_exception_if_processing_interrupted()

    try:
        with requests.post(
            "https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "search_depth": "basic",
                "max_results": max(1, min(10, int(max_results))),
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=(15, 60),
        ) as response:
            data = parse_json_response(response, "Tavily 搜索")

    except requests.Timeout:
        raise RuntimeError("联网搜索超时。") from None

    except requests.RequestException:
        raise RuntimeError("联网搜索连接失败。") from None

    mm.throw_exception_if_processing_interrupted()

    results = data.get("results") or []

    if not isinstance(results, list) or not results:
        raise RuntimeError(
            "搜索没有返回结果，请调整 search_query。"
        )

    blocks = []

    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue

        title = str(item.get("title") or "")[:500]
        url = str(item.get("url") or "")[:2000]
        content = str(item.get("content") or "")[:4000]

        blocks.append(
            f"[{index}]\n"
            f"标题：{title}\n"
            f"URL：{url}\n"
            f"摘要：{content}"
        )

    if not blocks:
        raise RuntimeError("搜索结果格式无效。")

    return "\n\n".join(blocks)


def model_names_from_payload(data):
    if isinstance(data, list):
        candidates = data
    elif isinstance(data, dict):
        candidates = None

        for key in (
            "data",
            "models",
            "result",
            "results",
        ):
            value = data.get(key)

            if isinstance(value, list):
                candidates = value
                break

            if isinstance(value, dict):
                nested = model_names_from_payload(value)

                if nested:
                    return nested
    else:
        return []

    if candidates is None:
        return []

    names = set()

    for item in candidates:
        if isinstance(item, str):
            name = item.strip()

        elif isinstance(item, dict):
            name = ""

            for key in ("id", "name", "model"):
                value = item.get(key)

                if isinstance(value, str) and value.strip():
                    name = value.strip()
                    break

        else:
            name = ""

        if name:
            names.add(name)

    return sorted(names, key=str.casefold)


def fallback_model_names(base_url):
    hostname = (
        urlsplit(base_url).hostname
        or ""
    ).casefold()

    return list(
        OFFICIAL_MODEL_FALLBACKS.get(
            hostname,
            (),
        )
    )


def fetch_model_names(
    base_url,
    api_key,
    direct=False,
    allow_direct_fallback=True,
):
    base_url = validate_base_url(base_url)

    headers = {}

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with request_api_endpoint(
            "GET",
            base_url,
            "/models",
            direct=direct,
            headers=headers,
            timeout=(10, 45),
        ) as response:
            data = parse_json_response(
                response,
                "模型列表",
                allow_list=True,
            )

    except requests.Timeout as error:
        if allow_direct_fallback and not direct:
            print(
                "[API LLM] 模型列表代理请求超时，"
                "禁用环境代理后直连重试一次。",
                flush=True,
            )
            return fetch_model_names(
                base_url,
                api_key,
                direct=True,
                allow_direct_fallback=False,
            )

        detail = sanitize_api_error(
            f"{type(error).__name__}: {error}"
        )
        message = "获取模型列表超时。"

        if detail:
            message += f"\n网络错误：{detail}"

        raise RuntimeError(message) from None

    except requests.RequestException as error:
        if allow_direct_fallback and not direct:
            print(
                "[API LLM] 模型列表代理连接异常，"
                "禁用环境代理后直连重试一次。",
                flush=True,
            )
            return fetch_model_names(
                base_url,
                api_key,
                direct=True,
                allow_direct_fallback=False,
            )

        detail = sanitize_api_error(
            f"{type(error).__name__}: {error}"
        )
        message = (
            "模型列表直连请求失败。"
            if direct
            else "获取模型列表连接失败。"
        )

        if detail:
            message += f"\n网络错误：{detail}"

        raise RuntimeError(message) from None

    models = model_names_from_payload(data)

    if not models:
        raise RuntimeError("接口没有返回可用模型。")

    return models


@PromptServer.instance.routes.post("/mini_api_llm/models")
async def refresh_models_route(request):
    try:
        data = await request.json()

        if not isinstance(data, dict):
            raise ValueError("请求必须是 JSON 对象。")

        base_url = validate_base_url(data.get("base_url", ""))
        api_key = str(data.get("api_key") or "").strip()

        # requests 放在线程中，避免阻塞 aiohttp 事件循环。
        models = await asyncio.to_thread(
            fetch_model_names,
            base_url,
            api_key,
        )

        return web.json_response({"models": models})

    except ValueError as error:
        return web.json_response(
            {"error": str(error)},
            status=400,
        )

    except RuntimeError as error:
        fallback_models = fallback_model_names(
            base_url
        )

        if fallback_models:
            return web.json_response({
                "models": fallback_models,
                "warning": (
                    "无法从接口读取模型列表，"
                    "当前显示内置候选模型。"
                    "手动输入的模型名仍然可以正常使用。\n\n"
                    f"{error}"
                ),
            })

        return web.json_response(
            {"error": str(error)},
            status=502,
        )

    except Exception:
        return web.json_response(
            {"error": "获取模型列表失败。"},
            status=500,
        )
