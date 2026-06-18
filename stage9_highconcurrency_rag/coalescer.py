"""
Request Coalescer — 请求合并器
================================
关键优化：当多个并发请求查询相同内容时，只执行一次，所有等待者共享结果。

场景：
  短时间内大量用户询问同一问题（热点问题、公告等）
  合并前：1000 个请求 → 1000 次 embedding + 1000 次向量搜索
  合并后：1000 个请求 → 1 次 embedding + 1 次向量搜索 → 1000 个响应

实现：Future 共享模式
  - 请求到来时先生成一个 Future
  - 如果相同 key 已有 inflight Future，直接 await 它
  - 第一个请求执行实际工作，完成后 set_result
"""

import asyncio
import hashlib
import time
from typing import Optional, Dict, Any, Callable, Awaitable


class RequestCoalescer:
    """
    请求合并器——相同 key 的并发请求只执行一次
    
    Example:
        coalescer = RequestCoalescer()
        
        async def expensive_op(key: str) -> str:
            return await some_work(key)
        
        # 100 个并发请求相同 key，只有 1 个实际执行
        results = await asyncio.gather(*[
            coalescer.execute("query1", expensive_op, "query1")
            for _ in range(100)
        ])
        # results 全部相同，expensive_op 只执行了 1 次
    """
    
    def __init__(self):
        self._inflight: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        
        # 统计
        self.total_requests = 0
        self.coalesced_requests = 0
        self.total_executions = 0
    
    async def execute(
        self,
        key: str,
        func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs,
    ) -> Any:
        """
        执行或合并请求
        
        如果 key 相同且正在执行中，等待已有结果。
        
        Args:
            key: 请求标识（通常为查询文本的 hash）
            func: 实际执行函数
            *args, **kwargs: 传递给 func 的参数
        
        Returns:
            执行结果
        """
        self.total_requests += 1
        
        async with self._lock:
            existing = self._inflight.get(key)
            if existing is not None:
                # 已有相同请求正在执行，合并
                self.coalesced_requests += 1
            else:
                # 新请求，创建 Future
                future = asyncio.get_event_loop().create_future()
                self._inflight[key] = future
                self.total_executions += 1
                existing = future
        
        # 等待结果
        if existing == self._inflight.get(key):
            # 我们是第一个请求，负责执行
            try:
                result = await func(*args, **kwargs)
                existing.set_result(result)
                return result
            except Exception as e:
                existing.set_exception(e)
                raise
            finally:
                async with self._lock:
                    self._inflight.pop(key, None)
        else:
            # 我们是合并者，等待第一个请求的结果
            return await existing
    
    def get_stats(self) -> dict:
        saved = self.total_requests - self.total_executions
        return {
            "total_requests": self.total_requests,
            "total_executions": self.total_executions,
            "coalesced": self.coalesced_requests,
            "saved_requests": saved,
            "merge_ratio": round(saved / self.total_requests, 4) if self.total_requests > 0 else 0,
        }


class QueryKey:
    """查询 key 生成工具"""
    
    @staticmethod
    def for_rag(query: str, top_k: int = 10) -> str:
        """生成 RAG 查询的缓存 key"""
        raw = f"rag:{query}:{top_k}"
        return hashlib.md5(raw.encode()).hexdigest()
    
    @staticmethod
    def for_llm(prompt: str, model: str, temperature: float = 0.0) -> str:
        """生成 LLM 调用的缓存 key"""
        raw = f"llm:{model}:{temperature}:{prompt}"
        return hashlib.md5(raw.encode()).hexdigest()
    
    @staticmethod
    def for_embedding(text: str) -> str:
        """生成 embedding 的缓存 key"""
        return hashlib.md5(f"emb:{text}".encode()).hexdigest()
