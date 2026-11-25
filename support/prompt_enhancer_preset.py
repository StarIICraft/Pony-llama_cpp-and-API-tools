from pathlib import Path


BUNDLED_PRESET_DIRECTORY = Path(__file__).with_name(
    "prompt_enhancer_presets"
)
PRESET_EXTENSIONS = {".txt", ".md"}


def _preset_suffix(filename):
    stem = Path(filename).stem
    if "_" not in stem:
        return stem
    return stem.split("_", 1)[1].lstrip()


def _ordered_files(directory):
    return sorted(
        (
            path
            for path in directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in PRESET_EXTENSIONS
            )
        ),
        key=lambda path: path.name.casefold(),
    )


def _discover_bundled_presets():
    if not BUNDLED_PRESET_DIRECTORY.is_dir():
        return {}

    files = _ordered_files(BUNDLED_PRESET_DIRECTORY)
    return {
        f"{index:02d}_{_preset_suffix(path.name)}": path
        for index, path in enumerate(files, start=1)
    }


def discover_prompt_enhancer_presets():
    return _discover_bundled_presets()


PRESET_FILES = discover_prompt_enhancer_presets()
PROMPT_ENHANCER_PRESET_NAMES = tuple(PRESET_FILES)
DEFAULT_PROMPT_ENHANCER_PRESET = (
    PROMPT_ENHANCER_PRESET_NAMES[0]
    if PROMPT_ENHANCER_PRESET_NAMES
    else ""
)


def get_prompt_enhancer_preset_names():
    return list(discover_prompt_enhancer_presets())


def read_prompt_enhancer_preset(name):
    path = discover_prompt_enhancer_presets().get(
        str(name or "")
    )
    if path is None:
        raise ValueError(
            f'Unknown prompt enhancer preset: "{name}"'
        )

    path = path.resolve()
    allowed_roots = {BUNDLED_PRESET_DIRECTORY.resolve()}
    if path.parent not in allowed_roots:
        raise ValueError("Invalid prompt enhancer preset path.")

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return file.read()
__all__ = [
    "DEFAULT_PROMPT_ENHANCER_PRESET",
    "PROMPT_ENHANCER_PRESET_NAMES",
    "discover_prompt_enhancer_presets",
    "get_prompt_enhancer_preset_names",
    "read_prompt_enhancer_preset",
]
