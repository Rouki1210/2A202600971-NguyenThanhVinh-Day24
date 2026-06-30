from __future__ import annotations

"""Module 2: Hybrid Search - BM25 Vietnamese + Dense + RRF."""

import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import (BM25_TOP_K, COLLECTION_NAME, DENSE_TOP_K, EMBEDDING_DIM,
                        EMBEDDING_MODEL, HYBRID_TOP_K, QDRANT_HOST, QDRANT_PORT)
except ModuleNotFoundError:
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    COLLECTION_NAME = "lab18_production"
    EMBEDDING_MODEL = "BAAI/bge-m3"
    EMBEDDING_DIM = 1024
    BM25_TOP_K = 20
    DENSE_TOP_K = 20
    HYBRID_TOP_K = 20


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words and normalize underthesea underscores."""
    try:
        from underthesea import word_tokenize

        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception:
        return text


def _tokenize(text: str) -> list[str]:
    segmented = segment_vietnamese(text).lower()
    return re.findall(r"\w+", segmented, flags=re.UNICODE)


class BM25Search:
    def __init__(self):
        self.corpus_tokens: list[list[str]] = []
        self.documents: list[dict] = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = [_tokenize(chunk.get("text", "")) for chunk in chunks]

        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.corpus_tokens)
        except Exception:
            self.bm25 = None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25, with a simple lexical fallback."""
        if not self.documents:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        if self.bm25 is not None:
            scores = self.bm25.get_scores(query_tokens)
        else:
            query_terms = set(query_tokens)
            scores = [
                sum(1 for token in doc_tokens if token in query_terms)
                for doc_tokens in self.corpus_tokens
            ]

        top_indices = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:top_k]
        results: list[SearchResult] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            doc = self.documents[idx]
            results.append(SearchResult(
                text=doc.get("text", ""),
                score=score,
                metadata=doc.get("metadata", {}),
                method="bm25",
            ))
        return results


class DenseSearch:
    def __init__(self):
        try:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        except Exception:
            self.client = None
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        if self.client is None or not chunks:
            return

        from qdrant_client.models import Distance, PointStruct, VectorParams

        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

        texts = [chunk.get("text", "") for chunk in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=True)
        points = [
            PointStruct(
                id=i,
                vector=vector.tolist(),
                payload={**chunk.get("metadata", {}), "text": chunk.get("text", "")},
            )
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if self.client is None:
            return []

        try:
            query_vector = self._get_encoder().encode(query).tolist()
            response = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=top_k,
            )
        except Exception:
            return []

        return [
            SearchResult(
                text=point.payload.get("text", ""),
                score=float(point.score),
                metadata={k: v for k, v in point.payload.items() if k != "text"},
                method="dense",
            )
            for point in response.points
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked result lists using reciprocal rank fusion."""
    fused: dict[str, dict] = {}

    for results in results_list:
        for rank, result in enumerate(results):
            if result.text not in fused:
                fused[result.text] = {"score": 0.0, "result": result}
            fused[result.text]["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
    merged: list[SearchResult] = []
    for item in ranked[:top_k]:
        result = item["result"]
        merged.append(SearchResult(
            text=result.text,
            score=float(item["score"]),
            metadata=result.metadata,
            method="hybrid",
        ))
    return merged


class HybridSearch:
    """Combines BM25 + Dense + RRF."""

    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    sample = "Nhan vien duoc nghi phep nam"
    print(f"Original:  {sample}")
    print(f"Segmented: {segment_vietnamese(sample)}")
