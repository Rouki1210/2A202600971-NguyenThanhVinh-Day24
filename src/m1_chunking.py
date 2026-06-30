from __future__ import annotations

"""Module 1: Advanced Chunking Strategies."""

import glob
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import (DATA_DIR, HIERARCHICAL_CHILD_SIZE, HIERARCHICAL_PARENT_SIZE,
                        SEMANTIC_THRESHOLD)
except ModuleNotFoundError:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    HIERARCHICAL_PARENT_SIZE = 2048
    HIERARCHICAL_CHILD_SIZE = 256
    SEMANTIC_THRESHOLD = 0.85


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer from a PDF."""
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError:
        return ""

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load markdown and text-layer PDFs from data/."""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  Skip {os.path.basename(fp)}: PDF has no text layer.")

    return docs


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """Baseline paragraph chunking."""
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """Group adjacent sentences by semantic similarity."""
    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n{2,}', text) if s.strip()]
    if not sentences:
        return []

    def to_chunks(groups: list[list[str]]) -> list[Chunk]:
        chunks = []
        for i, group in enumerate(groups):
            chunk_text = "\n\n".join(group).strip()
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={**metadata, "strategy": "semantic", "chunk_index": i},
                ))
        return chunks

    try:
        from sentence_transformers import SentenceTransformer
        from numpy import dot
        from numpy.linalg import norm

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(sentences)

        groups = [[sentences[0]]]
        for i in range(1, len(sentences)):
            sim = float(dot(embeddings[i - 1], embeddings[i]) / (norm(embeddings[i - 1]) * norm(embeddings[i]) + 1e-9))
            if sim < threshold:
                groups.append([sentences[i]])
            else:
                groups[-1].append(sentences[i])
        return to_chunks(groups)
    except Exception:
        # Offline fallback for environments without downloaded embedding models.
        groups = []
        current = []
        current_terms = set()
        for sentence in sentences:
            terms = set(re.findall(r"\w+", sentence.lower(), flags=re.UNICODE))
            overlap = len(current_terms & terms) / max(len(current_terms | terms), 1)
            if current and overlap < 0.08 and len(" ".join(current)) > 250:
                groups.append(current)
                current = []
                current_terms = set()
            current.append(sentence)
            current_terms |= terms
        if current:
            groups.append(current)
        return to_chunks(groups)


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """Create parent chunks and smaller child chunks linked by parent_id."""
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]

    parents: list[Chunk] = []
    current = ""
    for paragraph in paragraphs:
        separator = "\n\n" if current else ""
        if current and len(current) + len(separator) + len(paragraph) > parent_size:
            pid = f"parent_{len(parents)}"
            parents.append(Chunk(
                text=current.strip(),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": len(parents)},
            ))
            current = paragraph
        else:
            current = f"{current}{separator}{paragraph}"

    if current.strip():
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(
            text=current.strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": len(parents)},
        ))

    children: list[Chunk] = []
    for parent in parents:
        pid = parent.metadata["parent_id"]
        child_current = ""
        parts = [p.strip() for p in parent.text.split("\n\n") if p.strip()]

        for part in parts:
            pieces = [part]
            if len(part) > child_size:
                pieces = re.findall(r".{1,%d}(?:\s+|$)" % child_size, part, flags=re.DOTALL) or [part]

            for piece in pieces:
                piece = piece.strip()
                separator = "\n\n" if child_current else ""
                if child_current and len(child_current) + len(separator) + len(piece) > child_size:
                    children.append(Chunk(
                        text=child_current.strip(),
                        metadata={**metadata, "chunk_type": "child", "chunk_index": len(children), "parent_id": pid},
                        parent_id=pid,
                    ))
                    child_current = piece
                else:
                    child_current = f"{child_current}{separator}{piece}"

        if child_current.strip():
            children.append(Chunk(
                text=child_current.strip(),
                metadata={**metadata, "chunk_type": "child", "chunk_index": len(children), "parent_id": pid},
                parent_id=pid,
            ))

    return parents, children


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """Chunk markdown by heading sections while preserving headers."""
    metadata = metadata or {}
    chunks: list[Chunk] = []
    current_header = ""
    current_content: list[str] = []

    def flush() -> None:
        content = "\n".join(current_content).strip()
        if not current_header and not content:
            return
        chunk_text = f"{current_header}\n\n{content}".strip() if current_header else content
        section = current_header.lstrip("#").strip() if current_header else "root"
        chunks.append(Chunk(
            text=chunk_text,
            metadata={**metadata, "section": section, "strategy": "structure", "chunk_index": len(chunks)},
        ))

    for line in text.splitlines():
        if re.match(r"^#{1,3}\s+.+$", line):
            flush()
            current_header = line.strip()
            current_content = []
        else:
            current_content.append(line)

    flush()

    if not chunks and text.strip():
        chunks.append(Chunk(
            text=text.strip(),
            metadata={**metadata, "section": "root", "strategy": "structure", "chunk_index": 0},
        ))

    return chunks


def compare_strategies(documents: list[dict]) -> dict:
    """Run all strategies on documents and compare basic statistics."""
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
