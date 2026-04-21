"""
ingest.py
---------
Handles data loading from GitHub repos, chunking large documents,
and building the minsearch text index.
"""

import io
import re
import zipfile

import frontmatter
import requests
from minsearch import Index


# ---------------------------------------------------------------------------
# GitHub repo download
# ---------------------------------------------------------------------------

def read_repo_data(repo_owner: str, repo_name: str, branch: str = "main") -> list[dict]:
    """
    Download and parse all markdown (.md / .mdx) files from a GitHub repository.

    Args:
        repo_owner: GitHub username or organisation name.
        repo_name:  Repository name.
        branch:     Branch to download (default: "main").

    Returns:
        List of dicts with keys from frontmatter metadata + 'content' and 'filename'.
    """
    url = f"https://codeload.github.com/{repo_owner}/{repo_name}/zip/refs/heads/{branch}"
    print(f"Downloading {repo_owner}/{repo_name} …")
    resp = requests.get(url, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download repository: HTTP {resp.status_code}")

    repository_data: list[dict] = []
    zf = zipfile.ZipFile(io.BytesIO(resp.content))

    for file_info in zf.infolist():
        filename_lower = file_info.filename.lower()
        if not (filename_lower.endswith(".md") or filename_lower.endswith(".mdx")):
            continue

        try:
            with zf.open(file_info) as f_in:
                content = f_in.read().decode("utf-8", errors="ignore")
                post = frontmatter.loads(content)
                data = post.to_dict()

                # Strip the top-level zip-archive folder prefix (e.g. "repo-main/")
                parts = file_info.filename.split("/", maxsplit=1)
                data["filename"] = parts[1] if len(parts) == 2 else file_info.filename

                repository_data.append(data)
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠  Skipping {file_info.filename}: {exc}")
            continue

    zf.close()
    print(f"  ✓ {len(repository_data)} markdown files loaded.")
    return repository_data


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def sliding_window(text: str, size: int = 2000, step: int = 1000) -> list[dict]:
    """
    Split *text* into overlapping character-level chunks.

    Returns a list of dicts with keys 'content' and 'start'.
    """
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    result = []
    n = len(text)
    for i in range(0, n, step):
        chunk = text[i : i + size]
        result.append({"content": chunk, "start": i})
        if i + size >= n:
            break
    return result


def split_markdown_by_level(text: str, level: int = 2) -> list[str]:
    """
    Split markdown *text* by headers of the given *level*.

    Returns a list of section strings (each starts with the header line).
    """
    header_pattern = rf"^(#{{level}} )(.+)$".replace("{level}", str(level))
    pattern = re.compile(header_pattern, re.MULTILINE)
    parts = pattern.split(text)

    sections = []
    for i in range(1, len(parts), 3):
        header = (parts[i] + parts[i + 1]).strip()
        content = parts[i + 2].strip() if i + 2 < len(parts) else ""
        section = f"{header}\n\n{content}" if content else header
        sections.append(section)
    return sections


def chunk_documents(
    docs: list[dict],
    method: str = "sliding_window",
    size: int = 2000,
    step: int = 1000,
    level: int = 2,
) -> list[dict]:
    """
    Chunk a list of document dicts.

    Args:
        docs:   List of document dicts (must have a 'content' key).
        method: 'sliding_window' or 'sections'.
        size:   Chunk size in characters (sliding_window only).
        step:   Overlap step in characters (sliding_window only).
        level:  Markdown header level to split on (sections only).

    Returns:
        Flat list of chunk dicts preserving all original metadata.
    """
    chunks: list[dict] = []

    for doc in docs:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop("content", "")

        if method == "sections":
            raw_chunks = split_markdown_by_level(doc_content, level=level)
            for section in raw_chunks:
                chunk_doc = doc_copy.copy()
                chunk_doc["content"] = section
                chunks.append(chunk_doc)
        else:
            # default: sliding_window
            for chunk in sliding_window(doc_content, size=size, step=step):
                chunk_doc = doc_copy.copy()
                chunk_doc.update(chunk)  # adds 'content' and 'start'
                chunks.append(chunk_doc)

    return chunks


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def index_data(
    repo_owner: str,
    repo_name: str,
    branch: str = "main",
    filter_fn=None,
    chunk: bool = True,
    chunking_params: dict | None = None,
) -> tuple[Index, list[dict]]:
    """
    Download, optionally filter, chunk, and index repository data.

    Args:
        repo_owner:      GitHub owner.
        repo_name:       GitHub repo name.
        branch:          Branch name.
        filter_fn:       Optional callable(doc) -> bool to filter documents.
        chunk:           Whether to chunk large documents.
        chunking_params: Dict of kwargs forwarded to chunk_documents().

    Returns:
        (index, docs) — the fitted minsearch Index and the list of doc dicts.
    """
    docs = read_repo_data(repo_owner, repo_name, branch=branch)

    if filter_fn is not None:
        docs = [d for d in docs if filter_fn(d)]
        print(f"  ✓ {len(docs)} documents after filtering.")

    if chunk:
        params = chunking_params or {"method": "sliding_window", "size": 2000, "step": 1000}
        docs = chunk_documents(docs, **params)
        print(f"  ✓ {len(docs)} chunks created.")

    index = Index(text_fields=["content", "filename", "title", "description"])
    index.fit(docs)
    print(f"  ✓ Index built over {len(docs)} documents.")
    return index, docs
