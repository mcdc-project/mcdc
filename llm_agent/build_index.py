import json
import logging
import shutil
import ast
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re


from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import warnings

# This process may need to be revamped in the future

warnings.filterwarnings("ignore", message="could not convert string to float")
warnings.filterwarnings("ignore", category=UserWarning, module="pypdf")


# CONFIGURATION
REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = Path("llm_agent/scraped_docs")
INDEX_DIR = Path("llm_agent/vectorstore")
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# paths
RTD_DOCS_PATH = OUTPUT_DIR / "function_docs.json"
AUTO_API_PATH = OUTPUT_DIR / "auto_api.json"
EXAMPLES_DIR = OUTPUT_DIR / "examples"
PHYSICS_DOCS_DIR = Path("llm_agent/corpus/physics_docs")
SOURCE_CODE_DIR = REPO_ROOT / "mcdc"


# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)


def infer_section_from_function(func_name: str) -> str:
    """Map function name to workflow section."""
    func_lower = func_name.lower()

    if "material" in func_lower or "nuclide" in func_lower:
        return "material"
    elif (
        "surface" in func_lower
        or "plane" in func_lower
        or "cylinder" in func_lower
        or "sphere" in func_lower
    ):
        return "surface"
    elif "cell" in func_lower:
        return "cell"
    elif "universe" in func_lower or "lattice" in func_lower:
        return "hierarchy"
    elif "source" in func_lower:
        return "source"
    elif "tally" in func_lower or "mesh" in func_lower:
        return "tally"
    elif (
        "setting" in func_lower
        or "eigenmode" in func_lower
        or "particle" in func_lower
        or "batch" in func_lower
    ):
        return "settings"
    elif "run" in func_lower or "simulate" in func_lower:
        return "run"
    else:
        return "general"


# CHUNKING


def chunk_with_overlap(
    text: str,
    metadata: Dict[str, Any],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Smart chunking with overlap to maintain context.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(text)

    docs = []
    for i, chunk in enumerate(chunks):
        # Copy metadata and add chunk info
        chunk_metadata = metadata.copy()
        chunk_metadata.update(
            {"chunk_index": i, "total_chunks": len(chunks), "is_continuation": i > 0}
        )

        docs.append(Document(page_content=chunk, metadata=chunk_metadata))

    return docs


def extract_function_with_context(
    file_path: Path, node: ast.FunctionDef
) -> Tuple[str, Dict]:
    """
    Extract function with surrounding context.
    """
    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    # Get function code
    start_line = max(0, node.lineno - 1)
    end_line = node.end_lineno if node.end_lineno else start_line + 20

    code_lines = lines[start_line:end_line]
    code = "\n".join(code_lines)
    docstring = ast.get_docstring(node) or ""

    return code, {
        "docstring": docstring,
        "line_start": start_line + 1,
        "line_end": end_line,
        "lines": end_line - start_line,
    }


def chunk_code_semantically(file_path: Path, metadata_base: Dict) -> List[Document]:
    """
    Chunk code by logical units (functions/classes) with grouping.
    """
    docs = []

    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content)
        lines = content.splitlines()

        # Extract all top-level functions and classes
        nodes = [
            n
            for n in ast.iter_child_nodes(tree)
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        ]

        # Group small related functions
        i = 0
        while i < len(nodes):
            node = nodes[i]

            # Default section
            current_section = "general"

            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                current_section = infer_section_from_function(node.name)

            if isinstance(node, ast.FunctionDef):
                func_lines = (node.end_lineno or node.lineno) - node.lineno

                if func_lines < 50:
                    group = [node]
                    j = i + 1

                    # Group small functions
                    while j < len(nodes) and isinstance(nodes[j], ast.FunctionDef):
                        next_func = nodes[j]
                        next_lines = (
                            next_func.end_lineno or next_func.lineno
                        ) - next_func.lineno
                        if next_lines < 50 and len(group) < 3:
                            group.append(next_func)
                            j += 1
                        else:
                            break

                    # Create grouped document
                    start = group[0].lineno - 1
                    end = group[-1].end_lineno or group[-1].lineno

                    grouped_code = "\n".join(lines[start:end])
                    func_names = [f.name for f in group]

                    metadata = metadata_base.copy()
                    metadata.update(
                        {
                            "function_name": ", ".join(func_names),
                            "section": infer_section_from_function(func_names[0]),
                            "num_functions": len(group),
                            "line_start": start + 1,
                            "line_end": end,
                        }
                    )

                    docs.append(
                        Document(
                            page_content=f"Functions: {', '.join(func_names)}\n\n{grouped_code}",
                            metadata=metadata,
                        )
                    )

                    i = j

                elif func_lines < 150:
                    # Medium function - one per chunk
                    code, details = extract_function_with_context(file_path, node)

                    metadata = metadata_base.copy()
                    metadata.update(
                        {
                            "function_name": node.name,
                            "section": infer_section_from_function(node.name),
                            "num_functions": 1,
                            **details,
                        }
                    )

                    docs.append(Document(page_content=code, metadata=metadata))
                    i += 1

                else:
                    # Large function - split with overlap
                    start = node.lineno - 1
                    end = node.end_lineno or (start + 200)
                    full_code = "\n".join(lines[start:end])

                    metadata = metadata_base.copy()
                    metadata.update(
                        {
                            "function_name": node.name,
                            "section": infer_section_from_function(node.name),
                            "num_functions": 1,
                            "line_start": start + 1,
                            "line_end": end,
                            "is_large_function": True,
                        }
                    )

                    # Use chunking with overlap
                    chunks = chunk_with_overlap(full_code, metadata, chunk_size=800)
                    docs.extend(chunks)
                    i += 1

            elif isinstance(node, ast.ClassDef):
                # Classes: include definition + key methods
                start = node.lineno - 1
                end = node.end_lineno or (start + 100)
                class_code = "\n".join(lines[start:end])

                methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]

                metadata = metadata_base.copy()
                metadata.update(
                    {
                        "class_name": node.name,
                        "section": infer_section_from_function(node.name),  # ADDED
                        "methods": ", ".join(methods[:10]),
                        "num_methods": len(methods),
                        "line_start": start + 1,
                        "line_end": end,
                    }
                )

                if len(class_code) > 1500:
                    chunks = chunk_with_overlap(class_code, metadata, chunk_size=1000)
                    docs.extend(chunks)
                else:
                    docs.append(Document(page_content=class_code, metadata=metadata))

                i += 1
            else:
                i += 1  # Move to next node if not function or class

    except Exception as e:
        logging.debug(f"  Could not parse {file_path}: {e}")

    return docs


# SOURCE CODE LOADING


def load_source_code_enhanced() -> List[Document]:
    """
    Load source code with semantic chunking and context preservation.
    """
    docs = []

    if not SOURCE_CODE_DIR.exists():
        logging.warning(f"Source code directory not found at {SOURCE_CODE_DIR}")
        return docs

    logging.info(f"Loading source code with semantic chunking from {SOURCE_CODE_DIR}")

    py_files = list(SOURCE_CODE_DIR.rglob("*.py"))

    for py_file in py_files:
        module_path = py_file.relative_to(SOURCE_CODE_DIR.parent)
        module_name = str(module_path.with_suffix("")).replace("/", ".")

        module_is_private = is_private_module(py_file, SOURCE_CODE_DIR)

        metadata_base = {
            "source": "source_code" if not module_is_private else "internal_code",
            "type": "source_code" if not module_is_private else "internal_code",
            "module": module_name,
            "file_path": str(py_file.relative_to(REPO_ROOT)),
            "is_private": module_is_private,
            "quality": "high",
            "priority": 4 if not module_is_private else 2,
        }

        # Use semantic chunking (adds section metadata)
        file_docs = chunk_code_semantically(py_file, metadata_base)
        docs.extend(file_docs)

    logging.info(f"Loaded {len(docs)} semantic code chunks from {len(py_files)} files")
    return docs


def is_private_module(file_path: Path, source_root: Path) -> bool:
    """Determine if a module is private (internal implementation)."""
    if file_path.stem.startswith("_"):
        return True

    relative = file_path.relative_to(source_root)
    for part in relative.parts[:-1]:
        if part.startswith("_") or part == "test":
            return True

    internal_patterns = ["kernel", "loop", "adapter", "type_", "print_"]
    stem_lower = file_path.stem.lower()

    return any(pattern in stem_lower for pattern in internal_patterns)


# PDF LOADING (Updated)


def load_pdf_documents_enhanced() -> List[Document]:
    """
    Load PDFs page-by-page.
    """
    docs = []

    if not PHYSICS_DOCS_DIR.exists():
        logging.warning(f"Physics docs directory not found at {PHYSICS_DOCS_DIR}")
        return docs

    logging.info(f"Loading PDFs page-by-page from {PHYSICS_DOCS_DIR}")

    pdf_files = list(PHYSICS_DOCS_DIR.glob("*.pdf"))

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            paper_title = pdf_path.stem.replace("_", " ").title()

            for page_num, page in enumerate(pages, start=1):
                text = page.page_content.strip()

                if len(text) < 100:
                    continue

                content = f"Paper: {paper_title} (Page {page_num})\n\n{text}"

                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": "paper",
                            "type": "paper",
                            "section": "theory",
                            "paper_title": paper_title,
                            "page_number": page_num,
                            "file_path": str(
                                pdf_path.relative_to(PHYSICS_DOCS_DIR.parent)
                            ),
                            "quality": "high",
                            "priority": 5,
                        },
                    )
                )

            logging.info(f"Loaded {len(pages)} pages from {pdf_path.name}")

        except Exception as e:
            logging.error(f"Failed to load {pdf_path.name}: {e}")

    return docs


# EXAMPLE LOADING


def load_examples_enhanced() -> List[Document]:
    """
    Load examples, chunked by workflow section.
    """
    docs = []

    if not EXAMPLES_DIR.exists():
        logging.warning(f"Examples directory not found at {EXAMPLES_DIR}")
        return docs

    logging.info(f"Loading examples with section-based chunking from {EXAMPLES_DIR}")

    for example_file in EXAMPLES_DIR.glob("*.py"):
        content = example_file.read_text(encoding="utf-8")
        test_name = example_file.stem

        # Extract metadata
        complexity = infer_complexity(content, test_name)
        functions_used = list(set(re.findall(r"mcdc\.(\w+)", content)))

        # This helper finds all code blocks under # Material, # Surface, etc.
        workflow_sections = identify_workflow_sections(content)

        if not workflow_sections:
            # If no sections found, store whole example
            metadata = {
                "source": "example",
                "type": "full_example",
                "test_name": test_name,
                "section": "general",
                "complexity": complexity,
                "functions_used": ",".join(functions_used[:10]),
                "quality": "medium",
                "priority": 1,
            }
            docs.append(
                Document(
                    page_content=f"Example: {test_name} ({complexity})\n\n{content}",
                    metadata=metadata,
                )
            )
            continue

        # Create one document for each workflow section found
        for section_name, section_lines in workflow_sections.items():
            section_code = "\n".join(section_lines)

            metadata = {
                "source": "example",
                "type": "code_example",
                "test_name": test_name,
                "section": section_name,
                "complexity": complexity,
                "functions_used": ",".join(functions_used[:10]),
                "quality": "medium",
                "priority": 1,
            }

            docs.append(
                Document(
                    page_content=f"Example: {test_name} ({complexity})\nSection: {section_name.title()}\n\n{section_code}",
                    metadata=metadata,
                )
            )

    logging.info(f"Loaded {len(docs)} example section chunks")
    return docs


def identify_workflow_sections(content: str) -> Dict[str, List[str]]:
    """Identify workflow sections in example code."""
    sections = {}
    current_section = None
    lines = content.split("\n")

    # Section headers
    section_patterns = {
        "material": r"#.*[Ss]et materials?",
        "surface": r"#.*[Ss]et surfaces?",
        "cell": r"#.*[Ss]et cells?",
        "source": r"#.*[Ss]et source",
        "tally": r"#.*[Ss]et tallies?",
        "settings": r"#.*[Ss]ettings?",
        "run": r"#.*[Rr]un",
    }

    for line in lines:
        matched = None
        for section, pattern in section_patterns.items():
            if re.search(pattern, line, re.IGNORECASE):
                matched = section
                break

        # Add lines until we see another section header
        if matched:
            current_section = matched
            if current_section not in sections:
                sections[current_section] = []
            sections[current_section].append(line)
        elif current_section:
            sections[current_section].append(line)

    return sections


def infer_complexity(content: str, test_name: str) -> str:
    """Infer example complexity."""
    code_lines = [
        l for l in content.split("\n") if l.strip() and not l.strip().startswith("#")
    ]
    mcdc_calls = len(re.findall(r"mcdc\.\w+", content))

    has_advanced = any(k in content for k in ["Lattice", "Universe", "eigenmode"])

    if has_advanced or mcdc_calls > 25:
        return "advanced"
    elif mcdc_calls > 15 or len(code_lines) > 50:
        return "intermediate"
    return "beginner"


# API DOCS


def format_api_doc_readable(func_name: str, data: Dict[str, Any]) -> str:
    """Convert API doc to readable format."""
    parts = [f"Function: mcdc.{func_name}", "=" * 60]

    if "signature" in data:
        parts.append(f"\nSignature:\n{data['signature']}")
    if "description" in data:
        parts.append(f"\nDescription:\n{data['description']}")

    if "parameters" in data and data["parameters"]:
        parts.append("\nParameters:")
        for param in data["parameters"]:
            if isinstance(param, dict):
                name = param.get("name", "")
                desc = param.get("description", "")
                parts.append(f"  • {name}: {desc}")

    if "returns" in data:
        parts.append(f"\nReturns:\n{data['returns']}")

    return "\n".join(parts)


def load_api_docs() -> List[Document]:
    """
    Load API documentation from both RTD (curated) and auto-generated sources.
    """
    docs = []

    # RTD API Docs
    if RTD_DOCS_PATH.exists():
        logging.info(f"Loading RTD docs from {RTD_DOCS_PATH}")
        rtd_data = json.loads(RTD_DOCS_PATH.read_text(encoding="utf-8"))

        for func_name, data in rtd_data.items():
            readable_content = format_api_doc_readable(func_name, data)

            docs.append(
                Document(
                    page_content=readable_content,
                    metadata={
                        "source": "rtd",
                        "type": "api_doc",
                        "function": func_name,
                        "section": infer_section_from_function(func_name),
                        "quality": "high",
                        "priority": 5,
                    },
                )
            )
        logging.info(f"Loaded {len(rtd_data)} RTD docs")

    # Auto-generated API stubs
    if AUTO_API_PATH.exists():
        logging.info(f"Loading auto-generated docs from {AUTO_API_PATH}")
        auto_data = json.loads(AUTO_API_PATH.read_text(encoding="utf-8"))

        count = 0
        for qual_name, data in auto_data.items():
            # Only include top-level mcdc.*
            if qual_name.count(".") <= 2:
                func_name = qual_name.split(".")[-1]
                readable_content = format_api_doc_readable(func_name, data)

                docs.append(
                    Document(
                        page_content=readable_content,
                        metadata={
                            "source": "auto",
                            "type": "api_doc",
                            "function": func_name,
                            "qualified_name": qual_name,
                            "section": infer_section_from_function(func_name),
                            "quality": "medium",
                            "priority": 2,
                        },
                    )
                )
                count += 1
        logging.info(f"Loaded {count} auto-generated docs")

    return docs


# MAIN INDEXING


def load_all_documents() -> List[Document]:
    """Load all documents."""
    docs = []

    # API Docs
    docs.extend(load_api_docs())

    # Papers (page-by-page)
    docs.extend(load_pdf_documents_enhanced())

    # Source Code (semantic chunking)
    docs.extend(load_source_code_enhanced())

    # Examples (section chunking)
    docs.extend(load_examples_enhanced())

    logging.info(f"\n{'='*60}")
    logging.info(f"Total documents: {len(docs)}")
    logging.info(
        f"  - API docs: {sum(1 for d in docs if d.metadata.get('type') == 'api_doc')}"
    )
    logging.info(
        f"  - Papers: {sum(1 for d in docs if d.metadata.get('type') == 'paper')}"
    )
    logging.info(
        f"  - Source code: {sum(1 for d in docs if 'code' in d.metadata.get('type', ''))}"
    )
    logging.info(
        f"  - Examples: {sum(1 for d in docs if 'example' in d.metadata.get('type', ''))}"
    )
    logging.info(f"{'='*60}\n")

    return docs


def create_vectorstore(docs: List[Document]) -> Chroma:
    """Create fresh Chroma vectorstore."""
    if INDEX_DIR.exists():
        logging.info(f"Removing old index at {INDEX_DIR}")
        shutil.rmtree(INDEX_DIR)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    logging.info(f"Creating new vectorstore at {INDEX_DIR}")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(INDEX_DIR),
        collection_name="mcdc_docs",
        collection_metadata={"hnsw:space": "cosine"},
    )
    logging.info("Index created and saved")
    return vectorstore


def main():
    logging.info("=" * 60)
    logging.info("Building Enhanced RAG Index (with Section Tags)")
    logging.info("  - Semantic chunking with overlap")
    logging.info("  - Context-preserving code grouping")
    logging.info("  - Workflow-section example chunking")
    logging.info("=" * 60)

    docs = load_all_documents()
    if not docs:
        logging.error("No documents found")
        exit(1)

    create_vectorstore(docs)

    logging.info("\n" + "=" * 60)
    logging.info("Enhanced index build complete!")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
