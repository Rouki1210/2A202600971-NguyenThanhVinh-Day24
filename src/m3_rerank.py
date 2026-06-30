from __future__ import annotations

"""Module 3: Reranking - cross-encoder top-k reranking with fallback."""

import os
import re
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except ModuleNotFoundError:
                print("  sentence-transformers not installed; using lexical rerank fallback.")
                self._model = False
        return self._model

    @staticmethod
    def _fallback_score(query: str, document: str) -> float:
        query_terms = set(re.findall(r"\w+", query.lower(), flags=re.UNICODE))
        doc_terms = re.findall(r"\w+", document.lower(), flags=re.UNICODE)
        if not query_terms or not doc_terms:
            return 0.0

        overlap = sum(1 for term in doc_terms if term in query_terms)
        score = overlap / max(len(query_terms), 1)

        query_lower = query.lower()
        if re.search(r"\d+", document) and any(term in query_lower for term in ["bao", "nhi", "ngay"]):
            score += 0.25
        return score

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 -> top-k."""
        if not documents:
            return []

        model = self._load_model()
        if model:
            pairs = [(query, doc.get("text", "")) for doc in documents]
            scores = model.predict(pairs)
        else:
            scores = [self._fallback_score(query, doc.get("text", "")) for doc in documents]

        if isinstance(scores, (int, float)):
            scores = [scores]

        scored = sorted(zip(scores, documents), key=lambda item: float(item[0]), reverse=True)
        return [
            RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative placeholder."""

    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        return CrossEncoderReranker().rerank(query, documents, top_k=top_k)


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhan vien duoc nghi phep bao nhieu ngay?"
    docs = [
        {"text": "Nhan vien duoc nghi 12 ngay/nam.", "score": 0.8, "metadata": {}},
        {"text": "Mat khau thay doi moi 90 ngay.", "score": 0.7, "metadata": {}},
        {"text": "Thoi gian thu viec la 60 ngay.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
