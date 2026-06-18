"""
Embedding Service — 批处理引擎
================================
核心设计：
  1. 使用 asyncio 队列收集并发请求，达到 batch_size 或超时时触发批处理
  2. 线程池执行实际的 embedding 计算（避免阻塞事件循环）
  3. 支持本地模型（sentence-transformers）和远端 API 两种模式

吞吐量计算：
  - batch_size=64, batch_timeout=10ms
  - 每个 embedding ~1ms（ONNX 优化）
  - 理论单机 OPS: 64 / 0.010 = 6,400 (单线程) × N workers
"""

import asyncio
import time
import functools
import numpy as np
from typing import List, Optional, Callable, Awaitable


class EmbeddingService:
    """
    异步批处理 embedding 服务
    
    将并发请求聚合成 batch，减少模型调用次数，提高吞吐。
    
    Example:
        svc = EmbeddingService(model_name="all-MiniLM-L6-v2")
        await svc.start()
        vec = await svc.embed("hello world")
        vecs = await svc.embed_batch(["a", "b", "c"])
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        batch_size: int = 64,
        batch_timeout_ms: float = 10.0,
        max_workers: int = 4,
        use_local: bool = False,  # 默认为 mock/远端口
        remote_embed_fn: Optional[Callable[[List[str]], Awaitable[np.ndarray]]] = None,
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout_ms / 1000.0
        self.max_workers = max_workers
        self.use_local = use_local
        self._remote_embed_fn = remote_embed_fn
        
        # Batch 队列
        self._queue: asyncio.Queue = None
        self._pending: List[asyncio.Future] = None
        self._pending_lock: asyncio.Lock = None
        self._worker_task: asyncio.Task = None
        self._loop: asyncio.AbstractEventLoop = None
        
        # 统计
        self.batches_processed = 0
        self.total_embeddings = 0
        self.avg_batch_time_ms = 0.0
        
        # Lazy init for local model
        self._model = None
        self._tokenizer = None
        
    async def start(self):
        """启动批处理 worker"""
        self._loop = asyncio.get_event_loop()
        self._queue = asyncio.Queue()
        self._pending = []
        self._pending_lock = asyncio.Lock()
        self._worker_task = asyncio.create_task(self._batch_worker())
        
        if self.use_local:
            await self._init_local_model()
            
        print(f"[EmbeddingService] started | model={self.model_name} "
              f"dim={self.embedding_dim} batch_size={self.batch_size} "
              f"timeout={self.batch_timeout*1000:.0f}ms workers={self.max_workers}")
    
    async def stop(self):
        """优雅关闭"""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
    
    async def _init_local_model(self):
        """懒加载本地 embedding 模型——实际场景会在独立进程/GPU 上运行"""
        # 实际使用时替换为:
        # from sentence_transformers import SentenceTransformer
        # self._model = SentenceTransformer(self.model_name)
        # self._model.to("cuda")  # 如果有 GPU
        print(f"[EmbeddingService] Local model {self.model_name} initialized (mock)")
        self._model = "mock"
    
    async def embed(self, text: str) -> np.ndarray:
        """单条文本 embedding——通过批处理队列提交"""
        future = self._loop.create_future()
        async with self._pending_lock:
            self._pending.append((text, future))
        self._queue.put_nowait(None)  # 唤醒 worker
        return await future
    
    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量 embedding——立即执行，不走队列（用于预热/预计算）"""
        return await self._compute_embeddings(texts)
    
    async def _batch_worker(self):
        """
        批处理 worker 主循环
        
        设计要点：
        - 收到新请求后立即尝试收集更多，直到 batch_size 或超时
        - 超时策略保证延迟可控（最多等 batch_timeout）
        - 处理完后分发结果给所有等待者
        """
        while True:
            try:
                await asyncio.sleep(0)  # 让出控制权
                await self._queue.get()  # 等待至少一个请求
                
                async with self._pending_lock:
                    if not self._pending:
                        continue
                    batch = self._pending[:]
                    self._pending = []
                
                # 如果没满，等一会收集更多
                if len(batch) < self.batch_size:
                    await asyncio.sleep(self.batch_timeout)
                    async with self._pending_lock:
                        batch.extend(self._pending)
                        self._pending = []
                    # 只取 batch_size 个，剩下的留给下一轮
                    if len(batch) > self.batch_size:
                        async with self._pending_lock:
                            self._pending = batch[self.batch_size:]
                        batch = batch[:self.batch_size]
                
                t0 = time.perf_counter()
                texts = [item[0] for item in batch]
                futures = [item[1] for item in batch]
                
                vectors = await self._compute_embeddings(texts)
                
                t1 = time.perf_counter()
                elapsed_ms = (t1 - t0) * 1000
                
                # 分发结果
                for future, vec in zip(futures, vectors):
                    if not future.done():
                        future.set_result(vec)
                
                # 更新统计
                self.batches_processed += 1
                self.total_embeddings += len(batch)
                alpha = 0.9
                self.avg_batch_time_ms = alpha * self.avg_batch_time_ms + (1 - alpha) * elapsed_ms
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[EmbeddingService] batch error: {e}")
    
    async def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        核心计算——线程池执行实际 embedding
        
        本地模式：在 CPU/GPU 上通过 sentence-transformers 计算
        远端模式：通过 HTTP API 调用
        Mock 模式：返回随机向量（用于测试架构）
        """
        if self._remote_embed_fn:
            return await self._remote_embed_fn(texts)
        
        if self.use_local and self._model is not None:
            loop = asyncio.get_event_loop()
            fn = functools.partial(self._local_embed_sync, texts)
            return await loop.run_in_executor(None, fn)
        
        # Mock 模式：返回随机向量用于压力测试
        return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
    
    def _local_embed_sync(self, texts: List[str]) -> np.ndarray:
        """同步 embedding 计算（在线程池中执行）"""
        # 实际调用: return self._model.encode(texts, normalize_embeddings=True)
        return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
    
    def get_stats(self) -> dict:
        return {
            "batches_processed": self.batches_processed,
            "total_embeddings": self.total_embeddings,
            "avg_batch_time_ms": round(self.avg_batch_time_ms, 2),
            "batch_size": self.batch_size,
            "batch_timeout_ms": self.batch_timeout * 1000,
        }
