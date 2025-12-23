import os
import logging
from functools import lru_cache
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path

# Suppress verbose INFO logs from dependencies
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("google.generativeai").setLevel(logging.WARNING)

LLM_AGENT_DIR = Path(__file__).parent
DB_PATH = str(LLM_AGENT_DIR / "vectorstore")
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


# if encountering rate limits, use model="gemini-2.5-flash" or "gemini-2-flash"
def load_llm(temperature=0, model="gemini-3-flash-preview"):
    """Load Gemini LLM with API key validation."""
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found in environment variables. "
            "Please set it with: export GEMINI_API_KEY='your-api-key-here'"
        )

    return ChatGoogleGenerativeAI(
        model=model, temperature=temperature, google_api_key=api_key
    )


@lru_cache(maxsize=1)
def get_embeddings():
    """Cached embeddings model to avoid reloading."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"}
    )


def load_retriever(db_path: str = DB_PATH, k: int = 3, search_filter: dict = None):
    """
    Load retriever from ChromaDB.

    Supports optional metadata filtering.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Vector store not found at {db_path}. "
            "Please run your new build_index.py script."
        )

    embeddings = get_embeddings()

    vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)

    search_kwargs = {"k": k}
    if search_filter:
        search_kwargs["filter"] = search_filter

    return vectordb.as_retriever(search_kwargs=search_kwargs)


def format_docs(docs):
    """Format retrieved documents for RAG context."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        source = source.replace("llm_agent/corpus/", "")
        content = doc.page_content
        formatted.append(f"[Source {i}: {source}]\n{content}")

    return "\n\n---\n\n".join(formatted)


def create_rag_chain(retriever):
    """
    Chain that always searches docs before answering.
    Uses newer LangChain LCEL (LangChain Expression Language) syntax.
    """
    prompt = PromptTemplate.from_template(
        """You are MCDC-Agent. Answer based on docs.
    
Question: {question}

Docs: {context}

Answer:"""
    )

    # LCEL chain: retriever -> format -> prompt -> llm -> parse
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | load_llm()
        | StrOutputParser()
    )


def create_rag_chain_with_prompt(llm, retriever, prompt_template_str: str):
    """Create a RAG chain with custom prompt template."""
    prompt = PromptTemplate.from_template(prompt_template_str)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def create_qa_chain(llm, prompt_template_str: str):
    """Create a simple QA chain without retrieval."""
    prompt = PromptTemplate.from_template(prompt_template_str)

    qa_chain = prompt | llm | StrOutputParser()

    return qa_chain


# Shared functions for parsing LangChain agent responses
def extract_message_content(message) -> str:
    """
    Extract text content from a single message object.

    Handles various LangChain message formats including multimodal content blocks.
    """
    content = None
    if isinstance(message, dict):
        content = message.get("content", str(message))
    elif hasattr(message, "content"):
        content = message.content
    else:
        return str(message)

    if isinstance(content, list):
        # Handle multimodal content blocks - filter out empty ones
        text_parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                text = b.get("text", "").strip()
                if text:
                    text_parts.append(text)
            elif isinstance(b, str) and b.strip():
                text_parts.append(b)
        return "\n".join(text_parts)

    return str(content) if content else ""


def parse_agent_response(response) -> str:
    """
    Extract text from various LangChain response formats.

    Handles:
    - Plain strings
    - Dict responses with 'output', 'content', or 'messages' keys
    - Objects with .content attribute
    - List of content blocks
    """
    if isinstance(response, str):
        return response

    if isinstance(response, dict):
        if "output" in response:
            return response["output"]
        if "content" in response:
            return response["content"]
        if "messages" in response and response["messages"]:
            return extract_message_content(response["messages"][-1])

    if hasattr(response, "content"):
        return response.content

    if isinstance(response, list):
        try:
            text_parts = []
            for block in response:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif hasattr(block, "content"):
                    text_parts.append(block.content)
                else:
                    text_parts.append(str(block))
            return "\n".join(text_parts)
        except Exception:
            return str(response)

    return str(response)
