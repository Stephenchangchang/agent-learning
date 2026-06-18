"""Vector Store — FAISS (with pure NumPy fallback)"""

import os
import json
import time
import pickle
import asyncio
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


# Try FAISS, fall back to pure NumPy
try:
    import faiss as _faiss
    HAS_FAISS = True
except ImportError:
    _faiss = None
    HAS_FAISS = False
    print("[VectorStore] FAISS not installed; using NumPy brute-force fallback")


class FAISSVectorStore:
    """
    High-performance vector store with FAISS (NumPy fallback when unavailable).
    
    FAISS path: IVF index with HNSW coarse quantizer — O(log n) search.
    NumPy fallback: brute-force cosine similarity — O(n) search.
    """

    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "flat",
        nlist: int = 512,
        hnsw_m: int = 32,
        nprobe: int = 16,
        use_gpu: bool = False,
        metric: str = "cosine",
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.hnsw_m = hnsw_m
        self.nprobe = nprobe
        self.use_gpu = use_gpu and HAS_FAISS
        self.metric = metric
        self.has_faiss = HAS_FAISS

        self._index = None
        self._vectors: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self._documents: List[dict] = []
        self._doc_ids: List[str] = []

        self.total_searches = 0
        self.total_search_time_ms = 0.0
        self.avg_search_time_ms = 0.0

    def build(self, documents: List[dict], embeddings: np.ndarray):
        n = len(documents)
        if n == 0:
            return

        print(f"[VectorStore] Building index: {n} vectors, dim={self.dimension}", flush=True)
        t0 = time.perf_counter()
        vectors = embeddings.astype(np.float32)

        if self.has_faiss:
            self._build_faiss(vectors)
        else:
            self._build_numpy(vectors)

        # Store docs
        self._documents = documents
        self._doc_ids = [d["id"] for d in documents]

        elapsed = time.perf_counter() - t0
        engine = "FAISS" if self.has_faiss else "NumPy"
        print(f"[VectorStore] Index built in {elapsed*1000:.0f}ms | "
              f"engine={engine} type={self.index_type} vectors={n}")

    def _build_faiss(self, vectors: np.ndarray):
        """Build FAISS index (IVF or HNSW)."""
        import faiss
        d = self.dimension
        n = len(vectors)
        metric = faiss.METRIC_INNER_PRODUCT

        # Normalize for cosine
        faiss.normalize_L2(vectors)

        if self.index_type == "flat":
            self._index = faiss.IndexFlatIP(d)
            self._index.add(vectors)
        elif self.index_type == "ivf":
            nlist = min(self.nlist, max(1, n // 10))
            quantizer = faiss.IndexFlatIP(d)
            self._index = faiss.IndexIVFFlat(quantizer, d, nlist, metric)
            self._index.train(vectors)
            self._index.add(vectors)
            self._index.nprobe = self.nprobe
        elif self.index_type == "hnsw":
            self._index = faiss.IndexHNSWFlat(d, self.hnsw_m, metric)
            self._index.hnsw.efConstruction = 40
            self._index.add(vectors)

        # GPU
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self._index = faiss.index_cpu_to_gpu(res, 0, self._index)

    def _build_numpy(self, vectors: np.ndarray):
        """Store vectors for brute-force search (NumPy fallback)."""
        # Normalize for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._vectors = vectors / norms
        self._index = "numpy_fallback"

    def _search_faiss(self, query: np.ndarray, top_k: int) -> List[dict]:
        """Search using FAISS."""
        import faiss
        q = query.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)
        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(q, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            doc = self._documents[idx]
            score = max(0.0, min(1.0, float(dist) / 2 + 0.5))
            results.append({
                "id": doc["id"],
                "content": doc["content"],
                "score": round(score, 4),
                "metadata": doc.get("metadata", {}),
                "index": int(idx),
            })
        return results

    def _search_numpy(self, query: np.ndarray, top_k: int) -> List[dict]:
        """Brute-force cosine similarity using NumPy."""
        q = query.reshape(1, -1).astype(np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-10)
        scores = self._vectors @ q_norm.T  # (N, 1) dot product
        scores = scores.flatten()

        # Top-K
        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_scores = scores[top_indices]
        # Sort descending
        order = np.argsort(-top_scores)
        top_indices = top_indices[order]
        top_scores = top_scores[order]

        results = []
        for idx, score in zip(top_indices, top_scores):
            doc = self._documents[idx]
            score = max(0.0, min(1.0, float(score) / 2 + 0.5))
            results.append({
                "id": doc["id"],
                "content": doc["content"],
                "score": round(score, 4),
                "metadata": doc.get("metadata", {}),
                "index": int(idx),
            })
        return results

    def search_sync(self, query_vector: np.ndarray, top_k: int = 10) -> List[dict]:
        if self._index is None:
            return []

        t0 = time.perf_counter()

        if self.has_faiss:
            results = self._search_faiss(query_vector, top_k)
        else:
            results = self._search_numpy(query_vector, top_k)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        for r in results:
            r["latency_ms"] = round(elapsed_ms, 3)

        self.total_searches += 1
        self.total_search_time_ms += elapsed_ms
        alpha = 0.9
        self.avg_search_time_ms = alpha * self.avg_search_time_ms + (1 - alpha) * elapsed_ms

        return results

    async def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search_sync, query_vector, top_k)

    def add(self, documents: List[dict], embeddings: np.ndarray):
        if self._index is None:
            self.build(documents, embeddings)
            return

        vectors = embeddings.astype(np.float32)
        if self.has_faiss:
            import faiss
            faiss.normalize_L2(vectors)
            self._index.add(vectors)
        else:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._vectors = np.vstack([self._vectors, vectors / norms])

        for d in documents:
            self._documents.append(d)
            self._doc_ids.append(d["id"])

    @property
    def size(self) -> int:
        return len(self._documents)

    def get_stats(self) -> dict:
        return {
            "size": self.size,
            "index_type": self.index_type,
            "engine": "FAISS" if self.has_faiss else "NumPy",
            "dimension": self.dimension,
            "nprobe": self.nprobe,
            "total_searches": self.total_searches,
            "avg_search_time_ms": round(self.avg_search_time_ms, 3),
        }
