import ast, json
from pathlib import Path
from typing import Dict, List, Any, Set

REPO_ROOT = Path(__file__).parent.parent
MCDC_SOURCE = REPO_ROOT / "mcdc"
OUTPUT_JSON = Path("llm_agent/scraped_docs/auto_api.json")


def format_sig(node: ast.FunctionDef) -> str:
    """Reconstruct readable signature from AST."""
    args = []
    # positional args
    for arg in node.args.args:
        if arg.annotation:
            args.append(f"{arg.arg}: {ast.unparse(arg.annotation)}")
        else:
            args.append(arg.arg)
    # defaults (applied from right)
    if node.args.defaults:
        for i, d in enumerate(node.args.defaults):
            args[-len(node.args.defaults) + i] += f" = {ast.unparse(d)}"
    # *args
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    # **kwargs
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")
    return f"def {node.name}({', '.join(args)})"


def extract_from_node(node: ast.AST, pkg_name: str) -> Dict[str, Any]:
    """Get name, docstring, params from function or class."""
    data = {"name": node.name, "docstring": ast.get_docstring(node) or ""}

    if isinstance(node, ast.FunctionDef):
        data["type"] = "function"
        data["signature"] = format_sig(node)
        data["parameters"] = [
            {"name": arg.arg, "has_default": arg in node.args.defaults}
            for arg in node.args.args
        ]

    elif isinstance(node, ast.ClassDef):
        data["type"] = "class"
        # look at __init__ for constructor params
        init = next(
            (
                m
                for m in node.body
                if isinstance(m, ast.FunctionDef) and m.name == "__init__"
            ),
            None,
        )
        if init:
            data["init_signature"] = format_sig(init)
            data["init_parameters"] = [
                {"name": arg.arg, "has_default": arg in init.args.defaults}
                for arg in init.args.args[1:]  # skip 'self'
            ]
        else:
            data["init_signature"] = ""
            data["init_parameters"] = []
    return data


def walk_package(pkg_path: Path) -> Dict[str, Any]:
    """Walk every .py file in the package; extract all public functions/classes."""
    api: Dict[str, Any] = {}

    for py_file in pkg_path.rglob("*.py"):
        # Skip private modules and test files
        if any(part.startswith("_") for part in py_file.parts):
            continue

        module_name = ".".join(
            py_file.relative_to(pkg_path.parent).with_suffix("").parts
        )
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError:
            print(f"⚠  Skip {py_file} – parse error")
            continue

        # Module-level functions and classes
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                key = f"{module_name}.{node.name}"
                api[key] = extract_from_node(node, key)

    return api


def main():
    print(f"=== Walking {MCDC_SOURCE} ===")
    if not MCDC_SOURCE.exists():
        print(f"Error: {MCDC_SOURCE} not found")

    api = walk_package(MCDC_SOURCE)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(api, indent=2), encoding="utf-8")
    print(f"✓ Extracted {len(api)} symbols → {OUTPUT_JSON}")

    # Show what's missing from RTD
    rtd_funcs = {
        "material",
        "nuclide",
        "cell",
        "lattice",
        "surface",
        "universe",
        "eigenmode",
        "setting",
        "source",
        "tally",
    }
    auto_funcs = {
        k.split(".")[-1] for k in api.keys() if k.count(".") == 1
    }  # top-level only
    missing = auto_funcs - rtd_funcs
    if missing:
        print(f"\nMissing from RTD (but extracted from source): {sorted(missing)}")


if __name__ == "__main__":
    main()
