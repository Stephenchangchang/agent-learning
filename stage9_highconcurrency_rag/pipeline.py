"""
RAG Pipeline — 高性能异步流水线
===================================
设计理念：每个处理阶段都是独立的异步 worker，通过队列连接。

Pipeline 拓扑：
```
Query → [Router/Sharder] → [Cache Check] ──→ [Embedding]
                                    ↓            ↓
                                  [Return]    [Vector Search]
                                                  ↓
                                            [Rerank/Filter]
                                                  ↓
                              ┌──────────── [Cache Store] <────┐
                              ↓                               │
                          [LLM Generate] ──────────────────────┘
                              ↓
                          [Response]
```

每个阶段的特点：
  - 独立 worker 池：按阶段负载分配资源
  - 有界队列：防止背压崩溃，天然限流
  - 可插拔：每个阶段可以独立替换/升级
  - 可观测：每个阶段内置 metrics

背压策略：
  - 使用 asyncio.Queue(maxsize) 控制队列深度
  - 队列满时，上游 await put() 自然阻塞（背压传递）
  - 超时机制防止无限阻塞
"""

import os
import sys
import json
import time
import asyncio
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class Stage(Enum):
    PIPELINE_START = "start"
    CACHE_CHECK = "cache_check"
    EMBEDDING = "embedding"
    VECTOR_SEARCH = "vector_search"
    RERANK = "rerank"
    LLM_GENERATE = "llm_generate"
    CACHE_STORE = "cache_store"
    RESPONSE = "response"


@dataclass
class PipelineRequest:
    """流水线请求"""
    query: str
    request_id: str = ""
    top_k: int = 5
    use_llm: bool = True
    metadata: dict = field(default_factory=dict)
    created_at: float = 0.0
    
    # 流水线内部状态
    query_embedding: Optional[np.ndarray] = None
    retrieved_docs: List[dict] = field(default_factory=list)
    context: str = ""
    llm_response: str = ""
    cache_key: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()
        if not self.request_id:
            self.request_id = f"req_{int(self.created_at * 1_000_000)}_{id(self)}"


@dataclass
class PipelineResponse:
    """流水线响应"""
    request_id: str
    query: str
    context: str
    answer: str
    retrieved_docs: List[dict]
    stages_latency: dict  # stage_name -> ms
    total_latency_ms: float
    cache_hit: bool = False
    cache_level: str = "none"  # "l1" | "l2" | "l3" | "none"


class StageWorker:
    """
    流水线阶段 worker
    
    每个阶段运行独立的任务，从输入队列取请求，
    处理后放入下游队列。
    """
    
    def __init__(
        self,
        name: str,
        process_fn,
        num_workers: int = 4,
        max_queue_size: int = 1000,
    ):
        self.name = name
        self.process_fn = process_fn
        self.num_workers = num_workers
        self.input_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.output_queue: asyncio.Queue = None
        self._tasks: List[asyncio.Task] = []
        
        # 统计
        self.processed = 0
        self.total_time_ms = 0.0
        self.backpressure_count = 0
    
    def link_to(self, downstream: 'StageWorker'):
        """链接到下游 worker"""
        self.output_queue = downstream.input_queue
    
    async def run(self):
        """启动 N 个 worker 协程"""
        self._tasks = [
            asyncio.create_task(self._worker_loop(i))
            for i in range(self.num_workers)
        ]
    
    async def stop(self):
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
    
    async def _worker_loop(self, worker_id: int):
        while True:
            try:
                req: PipelineRequest = await self.input_queue.get()
                t0 = time.perf_counter()
                
                result = await self.process_fn(req)
                
                elapsed_ms = (time.perf_counter() - t0) * 1000
                self.processed += 1
                self.total_time_ms += elapsed_ms
                
                if self.output_queue is not None:
                    try:
                        await asyncio.wait_for(
                            self.output_queue.put(result),
                            timeout=1.0,
                        )
                    except asyncio.TimeoutError:
                        self.backpressure_count += 1
                        # 背压：丢弃请求或记录
                        if self.backpressure_count % 100 == 0:
                            print(f"[Pipeline] Backpressure at {self.name} "
                                  f"({self.backpressure_count} drops)")
                
                self.input_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Pipeline] Worker {self.name}/{worker_id} error: {e}")
    
    def get_stats(self) -> dict:
        avg_time = self.total_time_ms / self.processed if self.processed > 0 else 0
        return {
            "name": self.name,
            "num_workers": self.num_workers,
            "processed": self.processed,
            "avg_latency_ms": round(avg_time, 2),
            "backpressure_drops": self.backpressure_count,
            "queue_size": self.input_queue.qsize(),
            "queue_maxsize": self.input_queue.maxsize,
        }


class RAGPipeline:
    """
    高性能 RAG 流水线
    
    组装所有阶段，形成完整的处理链路。
    支持一键启动/停止和性能监控。
    
    Example:
        pipeline = RAGPipeline(
            embedding_svc=embedding_svc,
            vector_store=vector_store,
            cache=cache,
            llm=llm,
        )
        await pipeline.start()
        resp = await pipeline.process("What is RAG?")
        await pipeline.stop()
    """
    
    def __init__(
        self,
        embedding_svc=None,
        vector_store=None,
        cache=None,
        llm=None,
        coalescer=None,
        workers_per_stage: int = 4,
        max_queue_size: int = 1000,
    ):
        self.embedding_svc = embedding_svc
        self.vector_store = vector_store
        self.cache = cache
        self.llm = llm
        self.coalescer = coalescer
        self.workers_per_stage = workers_per_stage
        self.max_queue_size = max_queue_size
        
        self._workers: Dict[str, StageWorker] = {}
        self._direct_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._retriever_context_format = self._default_context_format
    
    def _default_context_format(self, docs: List[dict]) -> str:
        """将检索结果格式化为 LLM 上下文"""
        parts = []
        for i, doc in enumerate(docs):
            parts.append(f"[文档 {i+1}] (相关度: {doc.get('score', 0):.3f})\n{doc['content']}")
        return "\n\n".join(parts)
    
    async def start(self):
        """启动所有 worker"""
        self._running = True
        
        # 阶段1: 缓存检查
        self._cache_worker = StageWorker(
            "cache_check", self._cache_check_fn, 
            num_workers=self.workers_per_stage,
            max_queue_size=self.max_queue_size,
        )
        
        # 阶段2: Embedding
        self._embed_worker = StageWorker(
            "embedding", self._embedding_fn,
            num_workers=self.workers_per_stage,
            max_queue_size=self.max_queue_size,
        )
        
        # 阶段3: 向量检索
        self._search_worker = StageWorker(
            "vector_search", self._search_fn,
            num_workers=self.workers_per_stage,
            max_queue_size=self.max_queue_size,
        )
        
        # 阶段4: LLM 生成
        self._llm_worker = StageWorker(
            "llm_generate", self._llm_fn,
            num_workers=max(1, self.workers_per_stage // 2),
            max_queue_size=self.max_queue_size,
        )
        
        # 阶段5: 缓存写入
        self._store_worker = StageWorker(
            "cache_store", self._cache_store_fn,
            num_workers=self.workers_per_stage,
            max_queue_size=self.max_queue_size,
        )
        
        # 阶段6: 响应收集
        self._response_worker = StageWorker(
            "response", self._response_fn,
            num_workers=self.workers_per_stage,
            max_queue_size=self.max_queue_size * 10,  # 大一点，响应不回压
        )
        
        # 链接阶段：cache → embed → search → llm → store → response
        self._cache_worker.link_to(self._embed_worker)
        self._embed_worker.link_to(self._search_worker)
        self._search_worker.link_to(self._llm_worker)
        self._llm_worker.link_to(self._store_worker)
        self._store_worker.link_to(self._response_worker)
        
        # 启动所有 worker
        await asyncio.gather(
            self._cache_worker.run(),
            self._embed_worker.run(),
            self._search_worker.run(),
            self._llm_worker.run(),
            self._store_worker.run(),
            self._response_worker.run(),
        )
        
        print(f"[RAGPipeline] Started | {self.workers_per_stage} workers/stage "
              f"| queue_max={self.max_queue_size}")
    
    async def stop(self):
        self._running = False
        await asyncio.gather(
            self._cache_worker.stop(),
            self._embed_worker.stop(),
            self._search_worker.stop(),
            self._llm_worker.stop(),
            self._store_worker.stop(),
            self._response_worker.stop(),
        )
    
    async def process(self, query: str, **kwargs) -> PipelineResponse:
        """
        处理单个查询——走完整流水线
        
        实际场景中，这里收到请求后放入 cache_worker 的输入队列。
        但由于每个 worker 独立运行且队列解耦，complete 请求需要特殊处理。
        
        这里提供一个简化的直接调用路径（当前设计更清晰），
        生产环境通过队列解耦实现真正的高吞吐。
        """
        t0 = time.perf_counter()
        stages_latency = {}
        
        req = PipelineRequest(query=query, **kwargs)
        
        # Stage 1: Cache Check
        t1 = time.perf_counter()
        if self.cache:
            cache_key = f"rag:{query}:{req.top_k}"
            req.cache_key = cache_key
            for level, name in [(self.cache.l1, "l1"), (self.cache.l2, "l2"), (self.cache.l3, "l3")]:
                cached = level.get(cache_key) if level else None
                if cached is not None:
                    stages_latency["cache_check"] = (time.perf_counter() - t1) * 1000
                    return PipelineResponse(
                        request_id=req.request_id,
                        query=query,
                        context=cached.get("context", ""),
                        answer=cached.get("answer", ""),
                        retrieved_docs=cached.get("docs", []),
                        stages_latency=stages_latency,
                        total_latency_ms=(time.perf_counter() - t0) * 1000,
                        cache_hit=True,
                        cache_level=name,
                    )
        stages_latency["cache_check"] = (time.perf_counter() - t1) * 1000
        
        # Stage 2: Embedding
        t2 = time.perf_counter()
        if self.embedding_svc:
            if self.coalescer:
                req.query_embedding = await self.coalescer.execute(
                    f"emb:{query}", self.embedding_svc.embed, query
                )
            else:
                req.query_embedding = await self.embedding_svc.embed(query)
        stages_latency["embedding"] = (time.perf_counter() - t2) * 1000
        
        # Stage 3: Vector Search
        t3 = time.perf_counter()
        if self.vector_store is not None and req.query_embedding is not None:
            req.retrieved_docs = await self.vector_store.search(
                req.query_embedding, top_k=req.top_k
            )
            req.context = self._retriever_context_format(req.retrieved_docs)
        stages_latency["vector_search"] = (time.perf_counter() - t3) * 1000
        
        # Stage 4: LLM Generation
        t4 = time.perf_counter()
        if req.use_llm and self.llm and req.context:
            prompt = f"""基于以下检索到的文档回答用户问题。

检索到的文档：
{req.context}

用户问题：{query}

回答："""
            req.llm_response = await self.llm.generate(prompt)
        else:
            # 仅检索模式（不调用LLM）
            req.llm_response = req.context or "未找到相关信息"
        stages_latency["llm_generate"] = (time.perf_counter() - t4) * 1000
        
        # Stage 5: Cache Store (fire-and-forget)
        t5 = time.perf_counter()
        if self.cache and req.cache_key:
            self.cache.set(req.cache_key, {
                "context": req.context,
                "answer": req.llm_response,
                "docs": req.retrieved_docs,
            })
        stages_latency["cache_store"] = (time.perf_counter() - t5) * 1000
        
        total_ms = (time.perf_counter() - t0) * 1000
        
        return PipelineResponse(
            request_id=req.request_id,
            query=query,
            context=req.context,
            answer=req.llm_response,
            retrieved_docs=req.retrieved_docs,
            stages_latency={k: round(v, 2) for k, v in stages_latency.items()},
            total_latency_ms=round(total_ms, 2),
        )
    
    async def process_batch(self, queries: List[str], **kwargs) -> List[PipelineResponse]:
        """批量处理——并发执行"""
        tasks = [self.process(q, **kwargs) for q in queries]
        return await asyncio.gather(*tasks)
    
    # ============ Worker 处理函数 ============
    
    async def _cache_check_fn(self, req: PipelineRequest) -> PipelineRequest:
        """缓存检查阶段"""
        if self.cache:
            cache_key = f"rag:{req.query}:{req.top_k}"
            req.cache_key = cache_key
            for level in [self.cache.l1, self.cache.l2]:
                cached = level.get(cache_key)
                if cached is not None:
                    req.llm_response = cached.get("answer", "")
                    req.context = cached.get("context", "")
                    req.retrieved_docs = cached.get("docs", [])
        return req
    
    async def _embedding_fn(self, req: PipelineRequest) -> PipelineRequest:
        """Embedding 阶段"""
        if not req.llm_response and self.embedding_svc:
            if self.coalescer:
                req.query_embedding = await self.coalescer.execute(
                    f"emb:{req.query}", self.embedding_svc.embed, req.query
                )
            else:
                req.query_embedding = await self.embedding_svc.embed(req.query)
        return req
    
    async def _search_fn(self, req: PipelineRequest) -> PipelineRequest:
        """向量检索阶段"""
        if not req.llm_response and self.vector_store and req.query_embedding is not None:
            req.retrieved_docs = await self.vector_store.search(
                req.query_embedding, top_k=req.top_k
            )
            req.context = self._retriever_context_format(req.retrieved_docs)
        return req
    
    async def _llm_fn(self, req: PipelineRequest) -> PipelineRequest:
        """LLM 生成阶段"""
        if not req.llm_response and req.use_llm and self.llm and req.context:
            prompt = f"""基于以下检索到的文档回答用户问题。

检索到的文档：
{req.context}

用户问题：{req.query}

回答："""
            req.llm_response = await self.llm.generate(prompt)
        elif not req.llm_response:
            req.llm_response = req.context or "未找到相关信息"
        return req
    
    async def _cache_store_fn(self, req: PipelineRequest) -> PipelineRequest:
        """缓存写入阶段"""
        if self.cache and req.cache_key and req.llm_response:
            self.cache.set(req.cache_key, {
                "context": req.context,
                "answer": req.llm_response,
                "docs": req.retrieved_docs,
            })
        return req
    
    async def _response_fn(self, req: PipelineRequest) -> PipelineRequest:
        """响应收集阶段——计算延迟等"""
        return req
    
    def get_stats(self) -> dict:
        stats = {}
        for worker in [
            self._cache_worker, self._embed_worker, self._search_worker,
            self._llm_worker, self._store_worker, self._response_worker,
        ]:
            if worker:
                stats[worker.name] = worker.get_stats()
        return stats
