import importlib
import inspect
import sys
import types
from pathlib import Path


COMFY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMFY_ROOT))

from server import PromptServer  # noqa: E402


PromptServer.instance = types.SimpleNamespace(
    routes=types.SimpleNamespace(
        get=lambda *args, **kwargs: (lambda function: function),
        post=lambda *args, **kwargs: (lambda function: function),
    ),
)

package = importlib.import_module(
    "custom_nodes.Pony-llama_cpp-and-API-tools"
)


def declared_inputs(node_class):
    inputs = node_class.INPUT_TYPES()
    names = set()
    for section in ("required", "optional", "hidden"):
        names.update(inputs.get(section, {}))
    return names


def test_every_declared_input_reaches_node_function():
    failures = []
    input_count = 0

    for node_name, node_class in package.NODE_CLASS_MAPPINGS.items():
        function_name = getattr(node_class, "FUNCTION", None)
        if not function_name:
            continue
        function = getattr(node_class, function_name)
        parameters = inspect.signature(function).parameters.values()
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        accepted_names = {
            parameter.name
            for parameter in parameters
            if parameter.kind not in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            )
        }
        names = declared_inputs(node_class)
        input_count += len(names)
        missing = names - accepted_names
        if missing and not accepts_kwargs:
            failures.append((node_name, sorted(missing)))

    assert not failures, failures
    return len(package.NODE_CLASS_MAPPINGS), input_count


if __name__ == "__main__":
    node_count, input_count = (
        test_every_declared_input_reaches_node_function()
    )
    print(
        "external input routing: OK "
        f"({node_count} nodes, {input_count} declared inputs)"
    )
