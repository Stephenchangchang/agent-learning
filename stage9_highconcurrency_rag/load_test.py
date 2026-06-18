"""
Load Test & Benchmark — 负载测试和性能评估
============================================
设计：在本地模拟高并发 RAG 请求，测量吞吐量和延迟分布

测试策略：
  1. 渐增负载：逐步增加并发数，找到拐点
  2. 稳定负载：在目标 QPS 下运行 30s，验证稳定性
  3. 热点测试：高频相同 query，验证缓存在高并发下的效果
  4. 突发测试：瞬间爆发 1000+ 请求，验证背压和排队

输出指标：
  - Throughput (OPS): 每秒处理的请求数
  - P50/P95/P99 延迟: 延迟分布
  - 错误率: 超时/失败的请求比例
  - 缓存命中率: L1/L2 命中情况
  - 背压次数: Pipeline 各阶段的丢弃数
"""

import os
import sys
import time
import json
import asyncio
import random
import statistics
from typing import List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    name: str
    total_requests: int
    concurrency: int
    duration_sec: float
    throughput_ops: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_max_ms: float
    latency_min_ms: float
    error_count: int
    error_rate: float
    cache_hit_rate: float = 0.0
    component_stats: dict = field(default_factory=dict)
    
    def summary(self) -> str:
        lines = [
            f"=== {self.name} ===",
            f"  请求数: {self.total_requests}  |  并发: {self.concurrency}",
            f"  耗时: {self.duration_sec:.2f}s  |  吞吐: {self.throughput_ops:.0f} ops",
            f"  延迟: P50={self.latency_p50_ms:.1f}ms  P95={self.latency_p95_ms:.1f}ms  "
            f"P99={self.latency_p99_ms:.1f}ms  Max={self.latency_max_ms:.1f}ms",
            f"  错误: {self.error_count} ({self.error_rate*100:.1f}%)",
        ]
        if self.cache_hit_rate > 0:
            lines.append(f"  缓存命中: {self.cache_hit_rate*100:.1f}%")
        return "\n".join(lines)


class LoadTester:
    """
    负载测试器
    
    生成模拟查询，发送到 RAG Pipeline，收集性能数据。
    
    Example:
        tester = LoadTester(pipeline.process)
        
        # 测试不同并发级别
        for concurrency in [10, 50, 100, 500]:
            result = await tester.run_benchmark(
                name=f"concurrency_{concurrency}",
                num_queries=1000,
                concurrency=concurrency,
            )
            print(result.summary())
    """
    
    # 模拟查询池
    SAMPLE_QUERIES = [
        "什么是 RAG？",
        "如何提高检索准确率？",
        "向量数据库的优势是什么？",
        "什么是注意力机制？",
        "大模型的幻觉问题如何解决？",
        "Embedding 模型怎么选？",
        "什么是语义搜索？",
        "HNSW 索引的原理是什么？",
        "如何评估 RAG 系统？",
        "什么是查询改写？",
        "Reranker 的作用是什么？",
        "数据分块策略有哪些？",
        "多模态 RAG 怎么实现？",
        "什么是上下文窗口？",
        "Agent 系统中的工具调用机制",
        "如何构建知识图谱？",
        "稠密检索和稀疏检索的区别",
        "什么是端到端训练？",
        "大模型微调的方法总结",
        "量化技术对推理性能的影响",
    ]
    
    def __init__(
        self,
        process_fn: Callable[[str], Awaitable],
        queries_pool: Optional[List[str]] = None,
    ):
        self.process_fn = process_fn
        self.queries_pool = queries_pool or self.SAMPLE_QUERIES
    
    def generate_queries(self, count: int, hot_ratio: float = 0.0) -> List[str]:
        """
        生成测试查询
        
        hot_ratio: 热点查询比例（相同请求被重复发送，测试缓存效果）
        """
        queries = []
        hot_query = random.choice(self.queries_pool)
        for i in range(count):
            if random.random() < hot_ratio:
                queries.append(hot_query)
            else:
                queries.append(random.choice(self.queries_pool))
        return queries
    
    async def run_benchmark(
        self,
        name: str = "benchmark",
        num_queries: int = 1000,
        concurrency: int = 100,
        hot_ratio: float = 0.0,
        warmup: int = 100,
        timeout_ms: int = 10000,
    ) -> BenchmarkResult:
        """
        运行基准测试
        
        Args:
            name: 测试名称
            num_queries: 总请求数
            concurrency: 并发数（同时处理中的请求数）
            hot_ratio: 热点查询比例
            warmup: 预热请求数（不计入结果）
            timeout_ms: 单次请求超时
        """
        queries = self.generate_queries(num_queries + warmup, hot_ratio)
        
        # 预热
        if warmup > 0:
            warmup_queries = queries[:warmup]
            warmup_tasks = [self.process_fn(q) for q in warmup_queries]
            await asyncio.gather(*warmup_tasks, return_exceptions=True)
        
        test_queries = queries[warmup:]
        
        # 并发执行
        semaphore = asyncio.Semaphore(concurrency)
        latencies = []
        errors = 0
        
        async def _worker(query: str) -> float:
            async with semaphore:
                t0 = time.perf_counter()
                try:
                    resp = await asyncio.wait_for(
                        self.process_fn(query),
                        timeout=timeout_ms / 1000,
                    )
                    elapsed = (time.perf_counter() - t0) * 1000
                    return elapsed, resp
                except Exception as e:
                    elapsed = (time.perf_counter() - t0) * 1000
                    return elapsed, None
        
        t_start = time.perf_counter()
        tasks = [asyncio.create_task(_worker(q)) for q in test_queries]
        results = await asyncio.gather(*tasks)
        duration = time.perf_counter() - t_start
        
        cache_hits = 0
        for elapsed, resp in results:
            latencies.append(elapsed)
            if resp is None:
                errors += 1
            elif hasattr(resp, 'cache_hit') and resp.cache_hit:
                cache_hits += 1
        
        # 计算统计
        latencies.sort()
        n = len(latencies)
        p50 = latencies[int(n * 0.50)] if n > 0 else 0
        p95 = latencies[int(n * 0.95)] if n > 0 else 0
        p99 = latencies[int(n * 0.99)] if n > 0 else 0
        
        throughput = n / duration if duration > 0 else 0
        
        return BenchmarkResult(
            name=name,
            total_requests=n,
            concurrency=concurrency,
            duration_sec=duration,
            throughput_ops=throughput,
            latency_p50_ms=round(p50, 2),
            latency_p95_ms=round(p95, 2),
            latency_p99_ms=round(p99, 2),
            latency_max_ms=round(latencies[-1], 2) if latencies else 0,
            latency_min_ms=round(latencies[0], 2) if latencies else 0,
            error_count=errors,
            error_rate=errors / n if n > 0 else 0,
            cache_hit_rate=cache_hits / n if n > 0 else 0,
        )
    
    async def run_sweep(
        self,
        concurrency_levels: List[int] = None,
        num_queries_per_level: int = 1000,
        hot_ratios: List[float] = None,
    ) -> List[BenchmarkResult]:
        """
        遍历测试多个并发级别
        
        自动找到吞吐量拐点。
        """
        if concurrency_levels is None:
            concurrency_levels = [1, 5, 10, 25, 50, 100, 200, 500]
        if hot_ratios is None:
            hot_ratios = [0.0, 0.5]
        
        all_results = []
        for hot in hot_ratios:
            for c in concurrency_levels:
                result = await self.run_benchmark(
                    name=f"concurrency={c}_hot={hot}",
                    num_queries=num_queries_per_level,
                    concurrency=c,
                    hot_ratio=hot,
                    warmup=max(50, num_queries_per_level // 10),
                )
                all_results.append(result)
                print(result.summary())
                print()
        
        return all_results


def print_comparison(results: List[BenchmarkResult]):
    """打印对比表"""
    header = f"{'并发':>6} | {'吞吐(ops)':>10} | {'P50(ms)':>8} | {'P95(ms)':>8} | {'P99(ms)':>8} | {'错误率':>7} | {'缓存命中':>8}"
    sep = "-" * len(header)
    print(f"\n{'='*60}")
    print(f"性能对比汇总")
    print(f"{'='*60}")
    print(header)
    print(sep)
    for r in results:
        print(f"{r.concurrency:>6} | {r.throughput_ops:>10.0f} | {r.latency_p50_ms:>8.1f} | "
              f"{r.latency_p95_ms:>8.1f} | {r.latency_p99_ms:>8.1f} | {r.error_rate:>7.1%} | "
              f"{r.cache_hit_rate:>8.1%}")
    print(sep)
