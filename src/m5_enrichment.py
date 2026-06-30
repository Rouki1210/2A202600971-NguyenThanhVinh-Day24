from __future__ import annotations

"""Module 5: Chunk enrichment before embedding."""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import OPENAI_API_BASE, OPENAI_API_KEY, OPENAI_MODEL
except ModuleNotFoundError:
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", os.getenv("OPENAI_BASE_URL", ""))
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_OPENAI_ENRICHMENT_DISABLED = False


@dataclass
class EnrichedChunk:
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "metadata", "combined"


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _openai_chat(system: str, user: str, max_tokens: int = 200) -> str | None:
    global _OPENAI_ENRICHMENT_DISABLED

    if not OPENAI_API_KEY or _OPENAI_ENRICHMENT_DISABLED:
        return None

    try:
        from openai import OpenAI

        timeout = float(os.getenv("OPENAI_TIMEOUT", "8"))
        client_kwargs = {"api_key": OPENAI_API_KEY, "timeout": timeout}
        if OPENAI_API_BASE:
            client_kwargs["base_url"] = OPENAI_API_BASE
        client = OpenAI(**client_kwargs)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        _OPENAI_ENRICHMENT_DISABLED = True
        print(f"  OpenAI enrichment failed once; using local fallback for remaining chunks: {exc}")
        return None


def _strip_json_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned


def summarize_chunk(text: str) -> str:
    """Create a short Vietnamese summary for a chunk."""
    result = _openai_chat(
        "Tom tat doan van sau trong 1-2 cau ngan gon bang tieng Viet.",
        text,
        max_tokens=150,
    )
    if result:
        return result

    sentences = _sentences(text)
    if not sentences:
        return text.strip()
    return " ".join(sentences[:2]).strip()


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """Generate questions that this chunk can answer."""
    result = _openai_chat(
        f"Dua tren doan van, tao {n_questions} cau hoi ma doan van co the tra loi. Moi cau tren mot dong.",
        text,
        max_tokens=200,
    )
    if result:
        questions = [
            line.strip().lstrip("0123456789.-) ")
            for line in result.splitlines()
            if line.strip()
        ]
        return [q if q.endswith("?") else f"{q}?" for q in questions[:n_questions]]

    sentences = _sentences(text)
    questions: list[str] = []
    for sentence in sentences[:n_questions]:
        lowered = sentence.lower()
        if any(token in lowered for token in ["ngay", "ngay.", "12", "90", "60", "bao nhieu"]):
            questions.append("Thong tin nay quy dinh bao nhieu ngay?")
        elif any(token in lowered for token in ["mat khau", "mfa", "vpn"]):
            questions.append("Quy dinh cong nghe thong tin trong doan nay la gi?")
        elif any(token in lowered for token in ["nghi", "phep"]):
            questions.append("Nhan vien duoc nghi phep nhu the nao?")
        else:
            questions.append(f"Noi dung nay tra loi cau hoi gi ve: {sentence[:60]}?")

    if not questions and text.strip():
        questions.append("Doan van nay noi ve chinh sach gi?")
    return questions[:n_questions]


def contextual_prepend(text: str, document_title: str = "") -> str:
    """Prepend one context sentence while preserving the original chunk."""
    result = _openai_chat(
        "Viet 1 cau ngan mo ta doan van nay nam trong tai lieu nao va noi ve chu de gi. Chi tra ve 1 cau.",
        f"Tai lieu: {document_title}\n\nDoan van:\n{text}",
        max_tokens=80,
    )
    if result:
        return f"{result}\n\n{text}"

    if document_title:
        return f"Trich tu tai lieu {document_title}, doan nay cung cap thong tin lien quan den chinh sach noi bo.\n\n{text}"
    return f"Doan nay cung cap thong tin lien quan den chinh sach noi bo.\n\n{text}"


def extract_metadata(text: str) -> dict:
    """Extract lightweight metadata for retrieval filtering and reporting."""
    result = _openai_chat(
        'Trich xuat metadata tu doan van va chi tra ve JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance|safety|compliance|general", "language": "vi|en"}',
        text,
        max_tokens=180,
    )
    if result:
        try:
            parsed = json.loads(_strip_json_fence(result))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    lowered = text.lower()
    if any(term in lowered for term in ["mat khau", "mfa", "vpn", "malware", "cntt", "password"]):
        category = "it"
        topic = "information security"
    elif any(term in lowered for term in ["luong", "phu cap", "bao hiem", "vn"]):
        category = "finance"
        topic = "compensation and benefits"
    elif any(term in lowered for term in ["nghi", "phep", "thu viec", "nhan vien"]):
        category = "hr"
        topic = "employee policy"
    elif any(term in lowered for term in ["an toan", "pccc", "so cuu"]):
        category = "safety"
        topic = "workplace safety"
    else:
        category = "policy"
        topic = "general policy"

    entities = re.findall(r"\b[A-Z][A-Za-z0-9_.-]{1,}\b", text)
    return {
        "topic": topic,
        "entities": sorted(set(entities))[:10],
        "category": category,
        "language": "vi",
    }


def _enrich_single_call(text: str, source: str) -> dict:
    """Single call enrichment: summary, questions, context and metadata."""
    result = _openai_chat(
        """Analyze the Vietnamese chunk and return only JSON:
{
  "summary": "short 1-2 sentence summary",
  "questions": ["question 1", "question 2", "question 3"],
  "context": "one sentence explaining where this chunk fits in the document",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance|safety|compliance|general", "language": "vi|en"}
}""",
        f"Source: {source}\n\nChunk:\n{text}",
        max_tokens=400,
    )
    if result:
        try:
            parsed = json.loads(_strip_json_fence(result))
            if isinstance(parsed, dict):
                return {
                    "summary": str(parsed.get("summary", "")),
                    "questions": list(parsed.get("questions", [])),
                    "context": str(parsed.get("context", "")),
                    "metadata": dict(parsed.get("metadata", {})),
                }
        except (TypeError, json.JSONDecodeError):
            pass

    return {
        "summary": summarize_chunk(text),
        "questions": generate_hypothesis_questions(text, n_questions=3),
        "context": contextual_prepend(text, source).split("\n\n", 1)[0],
        "metadata": extract_metadata(text),
    }


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """Run enrichment over chunks."""
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods
    enriched: list[EnrichedChunk] = []

    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


if __name__ == "__main__":
    sample = (
        "Nhan vien chinh thuc duoc nghi phep nam 12 ngay lam viec moi nam. "
        "So ngay nghi phep tang them 1 ngay cho moi 5 nam tham nien cong tac."
    )

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")
    print(f"Summary: {summarize_chunk(sample)}\n")
    print(f"HyQA questions: {generate_hypothesis_questions(sample)}\n")
    print(f"Contextual: {contextual_prepend(sample, 'So tay nhan vien')}\n")
    print(f"Auto metadata: {extract_metadata(sample)}")
