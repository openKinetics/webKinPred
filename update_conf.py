import ast
import pathlib
import re
import sys

REQUIRED_EXTENSIONS = ("autoapi.extension", "sphinx.ext.napoleon")
AUTOAPI_DIRS_LINE = "autoapi_dirs = ['../autoapi_include']"
AUTOAPI_ADD_TOCTREE_LINE = "autoapi_add_toctree_entry = False"


def _append_extension(extensions: str, extension: str) -> str:
    """Compatibility helper retained for existing tests and callers."""
    if extensions.strip() == "[]":
        return f"['{extension}']"
    return extensions[:-1].rstrip() + f", '{extension}']"


def _format_extension_block(extensions: list[str]) -> str:
    """
    Formats a list of extensions into a string representation.

    Args:
        extensions (list[str]): A list of extension names as strings.

    Returns:
        str: A formatted string representing the list of extensions.

    """
    lines = ["extensions = ["]
    for extension in extensions:
        lines.append(f"    {extension!r},")
    lines.append("]")
    return "\n".join(lines)


def _replace_extensions_block(text: str) -> str:
    """
    Replace the 'extensions' assignment in a Python script with a merged list of required
    extensions.

    Args:
        text (str): The input Python script as a string.

    Returns:
        str: The modified script with updated 'extensions' assignment.

    """
    module = ast.parse(text)
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "extensions":
                    current_value = ast.literal_eval(node.value)
                    if not isinstance(current_value, list):
                        raise ValueError("docs/conf.py must define 'extensions' as a Python list.")
                    merged = list(current_value)
                    for extension in REQUIRED_EXTENSIONS:
                        if extension not in merged:
                            merged.append(extension)
                    replacement = _format_extension_block(merged)
                    lines = text.splitlines(keepends=True)
                    start = node.lineno - 1
                    end = node.end_lineno
                    updated = lines[:start] + [replacement + "\n"] + lines[end:]
                    return "".join(updated)
    return text.rstrip() + "\n\n" + _format_extension_block(list(REQUIRED_EXTENSIONS)) + "\n"


def update_conf(conf_py: str) -> None:
    """
    Update the configuration file by modifying its content if necessary.

    Args:
        conf_py (str): The path to the configuration file.

    Returns:
        None: This function does not return a value.

    """
    conf_path = pathlib.Path(conf_py)
    if not conf_path.exists():
        return

    original_text = conf_path.read_text(encoding="utf-8")
    text = _replace_extensions_block(original_text)
    if not re.search(r"^\s*autoapi_dirs\s*=", text, flags=re.MULTILINE):
        text = text.rstrip() + f"\n\n{AUTOAPI_DIRS_LINE}\n"
    if not re.search(r"^\s*autoapi_add_toctree_entry\s*=", text, flags=re.MULTILINE):
        text = text.rstrip() + f"\n{AUTOAPI_ADD_TOCTREE_LINE}\n"

    try:
        ast.parse(text)
    except SyntaxError as exc:
        raise ValueError(f"Updated docs/conf.py would be invalid Python: {exc.msg} (line {exc.lineno}).") from exc

    conf_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    update_conf(sys.argv[1])
