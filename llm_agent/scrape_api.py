import json, os, ast, sys
from pathlib import Path
from typing import Dict, List, Any, Set
import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).parent.parent
REGRESSION_DIR = REPO_ROOT / "test/regression"
OUTPUT_DIR = Path("llm_agent/scraped_docs")

# functions that actually have ReadTheDocs pages
RTD_FUNCTIONS = [
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
]

BASE_URL = "https://mcdc.readthedocs.io/en/stable/pythonapi/generated/mcdc.{}.html"

# UTILS


def _strip_classifier(text: str) -> str:
    """Remove '(type)' classifier so 'nuclides (list of tuple...)' → 'nuclides'"""
    return text.split("(")[0].strip()


def scrape_function_docs(function_name: str) -> Dict[str, Any]:
    """Scrape ONE RTD page. Robust against their HTML structure."""
    url = BASE_URL.format(function_name)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        return {"function": function_name, "error": str(e), "url": url}

    soup = BeautifulSoup(resp.text, "html.parser")

    # signature
    sig_dt = soup.find("dt", class_="sig sig-object py")
    signature = sig_dt.get_text(" ", strip=True) if sig_dt else ""

    # description – first paragraph after the title block
    desc_p = soup.select_one("section#mcdc-material p") or soup.select_one("section p")
    description = desc_p.get_text(" ", strip=True) if desc_p else ""

    # parameters
    params = []
    param_sect = soup.find("section", id="parameters")
    if param_sect:
        for dt, dd in zip(param_sect.find_all("dt"), param_sect.find_all("dd")):
            name = _strip_classifier(dt.get_text(" ", strip=True))
            desc = dd.get_text(" ", strip=True) if dd else ""
            params.append({"name": name, "description": desc})

    # returns
    returns = ""
    ret_sect = soup.find("section", id="returns")
    if ret_sect:
        dt, dd = ret_sect.find("dt"), ret_sect.find("dd")
        if dt and dd:
            returns = f"{dt.get_text(strip=True)}: {dd.get_text(' ', strip=True)}"

    return {
        "function": function_name,
        "signature": signature,
        "description": description,
        "parameters": params,
        "returns": returns,
        "url": url,
    }


# AST-BASED EXTRACTION  (catches *every* mcdc.* pattern)


def _get_full_attr_name(node: ast.AST) -> str:
    """
    Recursively unwraps nested ast.Attribute nodes to get the full
    dotted name, e.g., 'mcdc.settings.N_particle'.
    """
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return f"{_get_full_attr_name(node.value)}.{node.attr}"
    elif isinstance(node, ast.Call):
        return _get_full_attr_name(node.func)
    else:
        return ast.unparse(node)


def extract_mcdc_functions(code: str) -> Set[str]:
    """Return every distinct mcdc.X construct found in source."""
    found: Set[str] = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return found

    for node in ast.walk(tree):
        # mcdc.Function(...)  or  mcdc.Class(...)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            full_name = _get_full_attr_name(node.func)
            if full_name.startswith("mcdc."):
                found.add(full_name)

        # mcdc.settings.X = Y   assignments
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    full_name = _get_full_attr_name(target)
                    if full_name.startswith("mcdc."):
                        found.add(full_name)

    return {f for f in found if f.startswith("mcdc.")}


# regression tests extracting


def extract_examples_from_regression() -> List[Dict[str, Any]]:
    """Walk every regression test; return metadata + content."""
    examples = []
    if not REGRESSION_DIR.exists():
        print(f"⚠  {REGRESSION_DIR} not found – skipping examples")
        return examples

    for input_path in REGRESSION_DIR.rglob("input.py"):
        content = input_path.read_text(encoding="utf-8")
        lines = len(content.splitlines())
        funcs = sorted(extract_mcdc_functions(content))

        # Complexity tiers tuned to real MCDC tests
        complexity = (
            "beginner" if lines < 30 else "intermediate" if lines < 60 else "advanced"
        )

        examples.append(
            {
                "test_name": input_path.parent.name,
                "path": str(input_path.relative_to(REPO_ROOT)),
                "complexity": complexity,
                "lines": lines,
                "functions_used": funcs,
                "content": content,
            }
        )
    return examples


# SAVE HELPERS


def save_all(docs: Dict[str, Any], examples: List[Dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # RTD docs (10 functions)
    docs_path = OUTPUT_DIR / "function_docs.json"
    docs_path.write_text(json.dumps(docs, indent=2), encoding="utf-8")

    # Examples metadata
    jsonl_path = OUTPUT_DIR / "examples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fp:
        for ex in examples:
            fp.write(json.dumps(ex) + "\n")

    # Individual example files with header
    (OUTPUT_DIR / "examples").mkdir(exist_ok=True)
    for ex in examples:
        header = f'''"""Example: {ex["test_name"]}
Complexity: {ex["complexity"]}
Lines: {ex["lines"]}
Functions used: {", ".join(ex["functions_used"]) or "None"}
Path: {ex["path"]}
"""
'''
        (OUTPUT_DIR / "examples" / f"{ex['test_name']}.py").write_text(
            header + "\n" + ex["content"], encoding="utf-8"
        )

    print(f"✓ Saved {len(docs)} docs → {docs_path}")
    print(f"✓ Saved {len(examples)} examples → {jsonl_path} & examples/*.py")


# MAIN


def main() -> None:
    print("=== Scraping MCDC Docs (RTD + Regression) ===")
    docs = {}
    for func in RTD_FUNCTIONS:
        docs[func] = scrape_function_docs(func)
        status = "✗" if docs[func].get("error") else "✓"
        print(f"{status} {func}")

    print("\n=== Parsing Regression Tests ===")
    examples = extract_examples_from_regression()
    print(f"✓ Found {len(examples)} regression tests")

    print("\n=== Saving Data ===")
    save_all(docs, examples)

    ok = sum(1 for d in docs.values() if not d.get("error"))
    print(
        f"\n{'='*50}\nDone: {ok}/{len(docs)} docs, {len(examples)} examples\n{'='*50}"
    )


if __name__ == "__main__":
    main()
