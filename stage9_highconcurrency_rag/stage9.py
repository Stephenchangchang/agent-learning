#!/usr/bin/env python3
"""Stage 9: High-Concurrency RAG Agent — 目标: 50,000 ops"""

import os
import sys
import json
import time
import asyncio
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedding import EmbeddingService
from vector_store import FAISSVectorStore
from cache import MultiLevelCache
from coalescer import RequestCoalescer
from llm import LLMService, LLMConfig
from pipeline import RAGPipeline
from load_test import LoadTester, print_comparison


# ==================== 配置 ====================

OPS_TARGET = 50000

EMBEDDING_CONFIG = dict(
    model_name="all-MiniLM-L6-v2",
    embedding_dim=384,
    batch_size=64,
    batch_timeout_ms=10.0,
    max_workers=4,
)

VECTOR_STORE_CONFIG = dict(
    dimension=384,
    index_type="ivf",
    nlist=512,
    nprobe=16,
    metric="cosine",
)

CACHE_CONFIG = dict(
    l1_capacity=50000,
    l1_ttl_ms=5000,
    l2_db_path="/tmp/rag_bench_cache.sqlite",
    l2_ttl_ms=60000,
)

PIPELINE_CONFIG = dict(
    workers_per_stage=8,
    max_queue_size=2000,
)


# ==================== 知识库构建 ====================

def build_travel_knowledge_base(num_docs: int = 1000) -> list:
    """构建模拟旅游知识库"""
    topics = [
        "东京旅游攻略", "京都寺庙指南", "大阪美食推荐", "北海道冬季旅行",
        "巴黎景点介绍", "罗马历史文化", "曼谷夜市攻略", "新加坡滨海湾",
        "悉尼歌剧院", "巴厘岛度假指南", "马尔代夫潜水", "西安历史古迹",
        "成都美食攻略", "杭州西湖美景", "云南大理风光", "瑞士雪山滑雪",
        "冰岛极光之旅", "土耳其热气球", "新西兰自驾游", "伦敦博物馆",
        "纽约百老汇", "加拿大班芙", "埃及金字塔", "上海外滩夜景",
        "北京故宫历史", "日本温泉文化", "泰国清迈寺庙", "西班牙巴塞罗那",
    ]

    templates = [
        "{topic}是著名的旅游目的地，每年吸引大量游客前来观光。"
        "这里有着独特的文化和历史底蕴，游客可以欣赏到壮丽的自然风光。"
        "最佳旅游季节是春秋两季，气候宜人，适合户外活动。"
        "当地的美食也非常有特色，值得一试的包括传统小吃和特色菜肴。"
        "交通方面，公共交通系统发达，建议购买一日券或周游卡。",

        "前往{topic}旅行需要做好充分准备。首先，建议提前预订机票和酒店，"
        "旺季时价格会大幅上涨。当地货币兑换方便，各大商圈都设有兑换点。"
        "语言方面，英语和当地通用语可以满足基本沟通需求。"
        "特别提醒：注意保管个人财物，避免前往偏僻地区。",

        "{topic}以其独特的自然景观和人文风情闻名于世。"
        "这里的建筑风格融合了传统与现代元素，展现出独特的美学魅力。"
        "当地居民热情好客，游客可以深入体验当地的生活方式。"
        "推荐行程安排为3-5天，可以充分游览主要景点。",
    ]

    docs = []
    for i in range(num_docs):
        topic = random.choice(topics)
        template = random.choice(templates)
        content = template.format(topic=topic)
        docs.append({
            "id": f"doc_{i:06d}",
            "content": content,
            "metadata": {
                "topic": topic,
                "type": random.choice(["guide", "tips", "introduction"]),
                "length": len(content),
            },
        })
    return docs


# ==================== 吞吐量验证 ====================

async def validate_throughput(pipeline, target_ops: int, duration_sec: int = 5):
    """验证系统在目标吞吐量下的表现"""
    queries_pool = LoadTester.SAMPLE_QUERIES
    queries = [random.choice(queries_pool) for _ in range(target_ops * duration_sec)]

    print(f"\n{'='*60}")
    print(f"吞吐量验证: {target_ops:,} ops x {duration_sec}s")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    completed = 0
    errors = 0
    semaphore = asyncio.Semaphore(2000)

    async def worker(q):
        nonlocal completed, errors
        async with semaphore:
            try:
                await pipeline.process(q)
                completed += 1
            except Exception:
                errors += 1

    tasks = [asyncio.create_task(worker(q)) for q in queries]

    for sec in range(1, duration_sec + 1):
        await asyncio.sleep(1)
        current_ops = completed / sec
        print(f"  [{sec}/{duration_sec}s] {completed} done | {current_ops:.0f} ops | {errors} errors")

    await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.perf_counter() - t0
    actual_ops = completed / elapsed

    print(f"\nResult:")
    print(f"  Target:  {target_ops:,} ops")
    print(f"  Actual:  {actual_ops:.0f} ops")
    print(f"  {'PASS' if actual_ops >= target_ops else 'FAIL'} "
          f"({actual_ops/target_ops*100:.1f}%)")
    return actual_ops >= target_ops


# ==================== Demo ====================

async def run_demo(pipeline, vector_store):
    """交互式 demo"""
    print(f"\n{'='*60}")
    print(f"RAG Agent Demo")
    print(f"{'='*60}")
    print(f"KB: {vector_store.size:,} docs | Index: {vector_store.index_type}")
    print(f"Type 'quit' to exit\n")

    while True:
        try:
            query = input("? ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break

            t0 = time.perf_counter()
            resp = await pipeline.process(query, use_llm=False)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            print(f"\n[Results] ({elapsed_ms:.0f}ms) cache={resp.cache_hit}({resp.cache_level})")
            for i, doc in enumerate(resp.retrieved_docs[:3]):
                print(f"  {i+1}. [{doc['score']:.3f}] {doc['content'][:80]}...")
            print(f"[Stage latencies]")
            for stage, ms in resp.stages_latency.items():
                bar = chr(0x2588) * int(ms / 2)
                print(f"  {stage:20s} {ms:8.2f}ms {bar}")
            print()
        except KeyboardInterrupt:
            break


# ==================== 主入口 ====================

async def main():
    print(f"{'='*60}")
    print(f"Stage 9: High-Concurrency RAG Agent")
    print(f"Target: {OPS_TARGET:,} ops")
    print(f"{'='*60}\n")

    # 1. Embedding Service
    print("[1/6] Initializing Embedding Service...")
    embedding_svc = EmbeddingService(**EMBEDDING_CONFIG)
    await embedding_svc.start()

    # 2. Vector Store
    print("[2/6] Initializing Vector Store (FAISS)...")
    vector_store = FAISSVectorStore(**VECTOR_STORE_CONFIG)

    # 3. Build knowledge base
    print("[3/6] Building knowledge base...")
    num_docs = 100000
    docs = build_travel_knowledge_base(num_docs)
    print(f"  Generated {len(docs)} documents")

    # 4. Pre-compute embeddings
    print("[4/6] Pre-computing embeddings (batched)...")
    batch_size = 2000
    all_embeddings = []
    total_batches = (len(docs) + batch_size - 1) // batch_size
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        texts = [d["content"] for d in batch]
        embeddings = await embedding_svc.embed_batch(texts)
        all_embeddings.append(embeddings)
        done = min(i + batch_size, len(docs))
        print(f"  Batch {done}/{len(docs)}: {embeddings.shape}")

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"  Total embedding shape: {all_embeddings.shape}")

    # 5. Build FAISS index
    print("[5/6] Building FAISS index...")
    t0 = time.perf_counter()
    vector_store.build(docs, all_embeddings)
    t_build = time.perf_counter() - t0
    print(f"  Index built in {t_build*1000:.0f}ms")

    # 6. Initialize cache and pipeline
    print("[6/6] Initializing cache & pipeline...")
    cache = MultiLevelCache(**CACHE_CONFIG)
    coalescer = RequestCoalescer()
    pipeline = RAGPipeline(
        embedding_svc=embedding_svc,
        vector_store=vector_store,
        cache=cache,
        llm=None,
        coalescer=coalescer,
        workers_per_stage=PIPELINE_CONFIG["workers_per_stage"],
        max_queue_size=PIPELINE_CONFIG["max_queue_size"],
    )
    asyncio.create_task(pipeline.start())
    await asyncio.sleep(0.1)

    # ==== Warmup ====
    print("\nWarming up cache...")
    warmup_qs = [f"WU{i}: {q}" for i, q in enumerate(LoadTester.SAMPLE_QUERIES * 5)]
    await asyncio.gather(*[pipeline.process(q) for q in warmup_qs])
    print(f"  Cache ready: L1={cache.l1.get_stats()['size']}, L2={cache.l2.size}")

    # ==== Benchmark ====
    print(f"\n{'='*60}")
    print(f"Benchmark")
    print(f"{'='*60}\n")

    tester = LoadTester(pipeline.process)
    results = await tester.run_sweep(
        concurrency_levels=[1, 5, 10, 25, 50, 100, 200, 500, 1000],
        num_queries_per_level=2000,
        hot_ratios=[0.0],
    )
    print_comparison(results)

    # ==== Stats ====
    print(f"\n{'='*60}")
    print(f"Component Stats")
    print(f"{'='*60}")
    print(f"\nEmbedding: {json.dumps(embedding_svc.get_stats(), indent=2)}")
    print(f"\nVector Store: {json.dumps(vector_store.get_stats(), indent=2)}")
    print(f"\nCache: {json.dumps(cache.get_stats(), indent=2)}")
    print(f"\nCoalescer: {json.dumps(coalescer.get_stats(), indent=2)}")

    # ==== Throughput validation ====
    await validate_throughput(pipeline, target_ops=min(OPS_TARGET, 50000), duration_sec=5)

    # ==== Cleanup ====
    await pipeline.stop()
    await embedding_svc.stop()

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"  KB:    {vector_store.size:,} docs")
    print(f"  Index: {vector_store.index_type} / nprobe={vector_store.nprobe}")
    print(f"  Caches: L1={cache.l1.get_stats()['capacity']} / L2=SQLite WAL")
    print(f"  Pipeline: {PIPELINE_CONFIG['workers_per_stage']} workers/stage")
    print(f"  Coalescer: saved {coalescer.total_requests - coalescer.total_executions} dup reqs")
    print(f"\nNext steps to hit 50K ops:")
    print(f"  - GPU FAISS: vector search 0.5ms -> 0.01ms")
    print(f"  - GPU Embedding (CUDA): batch 10ms -> 0.5ms")
    print(f"  - Sharding: horizontal scale across nodes")
    print(f"  - vLLM local: remote 500ms -> local 10ms")
    print(f"  - Aggressive connection pool: 2000+ connections")


if __name__ == "__main__":
    asyncio.run(main())
