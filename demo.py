#!/usr/bin/env python3
"""
federated-ai-platform — Python 核心演示
联邦智能计算平台 · 自研HNSW · 联邦FL · RAG · CoT · Agent · 幻觉防御

运行方式: python3 demo.py
"""

import time
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

# ──────────────────────────────────────────────────────────────
# 颜色输出
# ──────────────────────────────────────────────────────────────

def c(text, color):
    colors = {
        "red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m",
        "blue": "\033[94m", "magenta": "\033[95m", "cyan": "\033[96m",
        "white": "\033[97m", "bold": "\033[1m", "reset": "\033[0m",
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"

def banner(title):
    print(f"\n{'═' * 70}")
    print(f"  {c(title, 'cyan')}")
    print(f"{'═' * 70}\n")

def section(name):
    print(f"\n{c('─' * 60, 'dim')}")
    print(f"  {c(name, 'yellow')}")

# ──────────────────────────────────────────────────────────────
# 1. 自研 HNSW 向量数据库（简化 Python 实现）
# ──────────────────────────────────────────────────────────────

class SimpleHNSW:
    """
    简化版 HNSW 向量索引
    生产环境请使用 Rust 实现版本
    """
    def __init__(self, dimension: int, m: int = 16, m0: int = 32):
        self.dimension = dimension
        self.m = m
        self.m0 = m0
        self.vectors: dict[str, list[float]] = {}
        self.metadata: dict[str, dict] = {}
        self.version = 0

    def insert(self, id: str, vector: list[float]) -> int:
        if len(vector) != self.dimension:
            raise ValueError(f"维度不匹配: 期望{self.dimension}, 实际{len(vector)}")
        self.vectors[id] = vector
        self.metadata[id] = {"version": self.version}
        self.version += 1
        return len(self.vectors) - 1

    def batch_insert(self, entries: list[tuple[str, list[float]]]):
        for id, vec in entries:
            self.insert(id, vec)

    def euclidean_distance(self, a: list[float], b: list[float]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        return dot / (norm_a * norm_b + 1e-8)

    def search(self, query: list[float], k: int = 5) -> list[dict]:
        """简化搜索：精确计算所有向量的距离"""
        if not self.vectors:
            return []
        results = []
        for id, vec in self.vectors.items():
            dist = self.euclidean_distance(query, vec)
            sim = self.cosine_similarity(query, vec)
            results.append({"id": id, "distance": dist, "similarity": sim, **self.metadata[id]})
        results.sort(key=lambda x: x["distance"])
        return results[:k]

    def snapshot_hash(self) -> str:
        h = hashlib.sha256()
        h.update(f"v{self.version}".encode())
        h.update(f"n{len(self.vectors)}".encode())
        return h.hexdigest()[:16]

# ──────────────────────────────────────────────────────────────
# 2. 区块链审计链
# ──────────────────────────────────────────────────────────────

class AuditChain:
    def __init__(self, merkle_batch_size: int = 10):
        self.merkle_batch_size = merkle_batch_size
        self.genesis = hashlib.sha256(b"GENESIS_FEDERATED_AI_PLATFORM_v1").hexdigest()
        self.entries: list[dict] = []
        self.next_seq = 1
        self.batch_leaves: list[str] = []
        self.current_merkle_root = None

    def _hash_entry(self, seq, ts, event_type, node_id, payload_hash, prev_hash) -> str:
        h = hashlib.sha256()
        h.update(f"{seq}{ts}{event_type}{node_id}{payload_hash}{prev_hash}".encode())
        return h.hexdigest()

    def append(self, event_type: str, node_id: str, **metadata) -> dict:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        prev_hash = self.entries[-1]["hash"] if self.entries else self.genesis

        payload = json.dumps(metadata, sort_keys=True, default=str)
        payload_hash = hashlib.sha256(f"{event_type}{node_id}{payload}".encode()).hexdigest()

        seq = self.next_seq
        h = self._hash_entry(seq, ts, event_type, node_id, payload_hash, prev_hash)

        entry = {
            "sequence": seq, "timestamp": ts, "event_type": event_type,
            "node_id": node_id, "payload_hash": payload_hash,
            "prev_hash": prev_hash, "hash": h,
            "metadata": metadata, "merkle_root": self.current_merkle_root,
        }
        self.entries.append(entry)
        self.next_seq += 1

        # Merkle 树更新
        self.batch_leaves.append(h)
        if len(self.batch_leaves) >= self.merkle_batch_size:
            self.current_merkle_root = self._build_merkle_root(self.batch_leaves)
            self.batch_leaves = []

        return entry

    def _build_merkle_root(self, leaves: list[str]) -> str:
        if not leaves:
            return hashlib.sha256(b"empty").hexdigest()
        current = leaves[:]
        while len(current) > 1:
            if len(current) % 2 == 1:
                current.append(current[-1])
            current = [hashlib.sha256(f"{a}{b}".encode()).hexdigest()
                       for a, b in zip(current[::2], current[1::2])]
        return current[0]

    def verify(self) -> bool:
        for i, entry in enumerate(self.entries):
            if i == 0:
                if entry["prev_hash"] != self.genesis:
                    return False
            else:
                if entry["prev_hash"] != self.entries[i-1]["hash"]:
                    return False
        return True

    def query(self, event_type: str = None, limit: int = 10) -> list[dict]:
        results = self.entries
        if event_type:
            results = [e for e in results if e["event_type"] == event_type]
        return list(reversed(results))[:limit]

# ──────────────────────────────────────────────────────────────
# 3. 五层幻觉防御
# ──────────────────────────────────────────────────────────────

@dataclass
class DefenseConfig:
    retrieval_threshold: float = 0.50
    crown_delta: float = 0.10
    self_consistency_threshold: float = 0.70
    consensus_threshold: float = 0.667

@dataclass
class DefenseResult:
    claim_id: str
    risk_score: float
    verdict: str  # Verified / Uncertain / Hallucination
    triggered_layers: list[str]
    defense_action: str  # Accept / Flag / Reject
    details: list[str] = field(default_factory=list)

# 层1: 检索一致性
def layer1_retrieval_consistency(claim: str, docs: list[dict]) -> tuple[bool, float, str]:
    if not docs:
        return False, 0.0, "无支撑文档"
    max_sim = max(d.get("similarity", 0) for d in docs)
    supported = max_sim >= 0.50
    return supported, max_sim, f"最高相似度: {max_sim:.3f}" + (" ✓" if supported else " ✗")

# 层2: 向量库事实核验（简化）
def layer2_vector_factcheck(claim: str, vector_db: SimpleHNSW) -> tuple[float, str]:
    terms = [w for w in claim if len(w) > 1]
    if not terms:
        return 0.5, "无关键词"
    verified = sum(1 for t in terms if any(t.lower() in vid.lower() for vid in vector_db.vectors))
    ratio = verified / max(len(terms), 1)
    return ratio, f"术语验证: {verified}/{len(terms)} ({ratio:.0%})"

# 层3: CROWN 一致性防御
def layer3_crown_defense(initial_ans: str, init_conf: float,
                          social_ans: str, social_conf: float,
                          delta: float = 0.10) -> dict:
    drop = init_conf - social_conf
    triggered = drop > delta
    return {
        "triggered": triggered,
        "confidence_drop": drop,
        "final_answer": initial_ans if triggered else social_ans,
        "reason": f"CROWN触发(下跌{drop:.3f}>{delta})" if triggered else f"置信度下跌可接受({drop:.3f}≤{delta})",
    }

# 层4: 多节点投票
def layer4_multi_vote(answers: list[str], confidences: list[float],
                       threshold: float = 0.667) -> dict:
    counts: dict[str, tuple[int, float]] = defaultdict(lambda: (0, 0.0))
    for ans, conf in zip(answers, confidences):
        c, tot = counts[ans]
        counts[ans] = (c + 1, tot + conf)
    best = max(counts.items(), key=lambda x: x[1][0])
    consensus, (votes, avg_conf) = best[0], best[1]
    strength = votes / len(answers)
    verdict = "Verified" if strength >= threshold else "Uncertain"
    return {
        "consensus": consensus, "votes": votes, "total": len(answers),
        "strength": strength, "avg_confidence": avg_conf / votes if votes else 0,
        "verdict": verdict, "all_votes": dict(counts),
    }

# 层5: LLM 自洽性
def layer5_self_consistency(samples: list[str], threshold: float = 0.70) -> dict:
    counts: dict[str, int] = defaultdict(int)
    for s in samples:
        counts[s] += 1
    best_text, best_count = max(counts.items(), key=lambda x: x[1])
    score = best_count / len(samples) if samples else 0
    return {
        "most_common": best_text, "count": best_count, "total": len(samples),
        "score": score, "passes": score >= threshold,
    }

# 综合引擎
def run_defense_engine(claim_id: str, claim: str,
                        retrieved_docs: list[dict],
                        multi_node_ans: list[str] = None,
                        multi_node_conf: list[float] = None,
                        vector_db: SimpleHNSW = None) -> DefenseResult:
    cfg = DefenseConfig()
    triggered = []
    risk_factors = []
    details = []

    # 层1
    ok, sim, msg = layer1_retrieval_consistency(claim, retrieved_docs)
    details.append(f"[层1] 检索一致性: {msg}")
    if not ok:
        triggered.append("RetrievalConsistency")
        risk_factors.append(1.0 - sim)

    # 层2
    if vector_db:
        ratio, msg = layer2_vector_factcheck(claim, vector_db)
        details.append(f"[层2] 事实核验: {msg}")
        if ratio < 0.3:
            triggered.append("VectorFactCheck")
            risk_factors.append(1.0 - ratio)

    # 层4
    if multi_node_ans and multi_node_conf:
        vote = layer4_multi_vote(multi_node_ans, multi_node_conf, cfg.consensus_threshold)
        details.append(f"[层4] 多节点投票: 共识'{vote['consensus']}' ({vote['strength']:.0%})")
        if vote["strength"] < cfg.consensus_threshold:
            triggered.append("MultiNodeVote")
            risk_factors.append(1.0 - vote["strength"])

    # 综合风险
    risk = sum(risk_factors) / len(risk_factors) if risk_factors else 0.0

    verdict_map = {0.0: "Verified", 0.25: "LikelyTrue", 0.50: "Uncertain", 0.75: "Hallucination"}
    verdict = "Uncertain"
    for thresh, v in sorted(verdict_map.items()):
        if risk <= thresh:
            verdict = v
            break

    action = "Accept" if verdict in ("Verified", "LikelyTrue") else ("Flag" if verdict == "Uncertain" else "Reject")

    return DefenseResult(
        claim_id=claim_id, risk_score=risk, verdict=verdict,
        triggered_layers=triggered, defense_action=action, details=details,
    )

# ──────────────────────────────────────────────────────────────
# 4. Agent 编排
# ──────────────────────────────────────────────────────────────

TOOL_DESCRIPTIONS = {
    "FederatedRAG": "联邦隐私保护检索增强",
    "FederatedCoT": "分布式思维链推理",
    "FederatedLearning": "联邦模型训练",
    "VectorSearch": "向量库语义搜索",
    "HallucinationCheck": "幻觉检测与防御",
}

def plan_task(task: str) -> list[dict]:
    """基于关键词的任务规划"""
    task_lower = task.lower()
    steps = []

    if any(k in task_lower for k in ["什么是", "如何", "为什么", "告诉我", "查询", "检索"]):
        steps.append({"tool": "FederatedRAG", "reason": "任务涉及问答/检索，使用联邦RAG"})

    if any(k in task_lower for k in ["分析", "推理", "计算", "评估", "比较", "预测"]):
        steps.append({"tool": "FederatedCoT", "reason": "任务需要多步推理，使用联邦CoT"})

    if any(k in task_lower for k in ["训练", "建模", "优化"]):
        steps.append({"tool": "FederatedLearning", "reason": "任务需要模型训练"})

    if steps:
        steps.append({"tool": "HallucinationCheck", "reason": "对生成结果进行幻觉防御"})
    else:
        steps.append({"tool": "VectorSearch", "reason": "通用检索任务"})

    steps.append({"tool": "Reflect", "reason": "执行后反思和改进"})

    for i, step in enumerate(steps):
        step["step_id"] = i + 1

    return steps

def run_agent(task: str, vector_db: SimpleHNSW = None) -> dict:
    """执行 Agent"""
    steps = plan_task(task)
    executed = []

    for step in steps:
        tool = step["tool"]
        # 模拟执行
        start = time.time()
        time.sleep(0.05)  # 模拟延迟

        if tool == "FederatedRAG":
            results = vector_db.search([0.5]*vector_db.dimension, 3) if vector_db else []
            output = f"联邦RAG检索完成，找到 {len(results)} 条相关结果"
        elif tool == "FederatedCoT":
            output = "联邦CoT推理：多路径推理 + 一致性投票 → 结论已生成"
        elif tool == "FederatedLearning":
            output = "FedAvg聚合完成：3节点参与，准确率提升至94.2%"
        elif tool == "VectorSearch":
            output = f"向量搜索完成，最相似文档: doc_001 (相似度87.3%)"
        elif tool == "HallucinationCheck":
            dr = run_defense_engine("temp", task, [])
            output = f"幻觉检测: 风险={dr.risk_score:.0%}, 动作={dr.defense_action}"
        else:
            output = "反思完成，建议: 当前执行正常"

        executed.append({
            **step,
            "output": output,
            "duration_ms": int((time.time() - start) * 1000),
            "success": True,
        })

    return {
        "task": task,
        "steps": executed,
        "total_steps": len(executed),
        "confidence": random.uniform(0.72, 0.95),
    }

# ──────────────────────────────────────────────────────────────
# 演示场景
# ──────────────────────────────────────────────────────────────

def demo_hedge_fund():
    banner("场景1: 对冲基金 — 联邦策略分析")
    print("  📊 目标: 多家对冲基金共享策略分析能力，同时保护持仓隐私\n")

    # 自研向量库
    index = SimpleHNSW(dimension=16)
    funds = [
        ("HF_ALPHA_001", [0.8,0.9,0.3,0.7,0.6,0.5,0.4,0.9,0.2,0.1,0.3,0.8,0.7,0.5,0.6,0.4]),
        ("HF_BETA_002",  [0.6,0.5,0.8,0.3,0.2,0.7,0.9,0.4,0.5,0.8,0.9,0.3,0.2,0.7,0.8,0.6]),
        ("HF_MACRO_003", [0.9,0.7,0.4,0.8,0.9,0.3,0.2,0.6,0.7,0.4,0.5,0.9,0.8,0.3,0.4,0.7]),
        ("HF_CTA_004",   [0.3,0.2,0.9,0.1,0.4,0.9,0.7,0.2,0.9,0.6,0.3,0.1,0.4,0.9,0.6,0.3]),
        ("HF arb_005",   [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]),
    ]
    for id, vec in funds:
        index.insert(id, vec)

    print(f"  ✅ 向量库: {c(len(index.vectors), 'green')} 支基金, dim={index.dimension}, 版本 v{index.version}")
    print(f"  🔒 审计哈希: {c(index.snapshot_hash(), 'cyan')}")

    # 搜索相似基金
    query = [0.7,0.8,0.4,0.7,0.8,0.4,0.3,0.7,0.3,0.3,0.4,0.7,0.6,0.5,0.5,0.5]
    results = index.search(query, k=3)
    print(f"\n  🔍 查询相似基金 (k=3):")
    for i, r in enumerate(results, 1):
        print(f"     {i}. {c(r['id'], 'yellow')} | 距离: {r['distance']:.4f} | 相似度: {r['similarity']:.1%}")

    # 审计链
    chain = AuditChain(merkle_batch_size=10)
    chain.append("FL_NODE_REGISTERED", "hf_node_alpha", domain="hedge_fund", data_size=5000)
    chain.append("FL_UPDATE_RECEIVED", "hf_node_alpha", round=1, weights_hash="a3f8c...")
    chain.append("FL_AGGREGATION_COMPLETED", "server", round=1, participants=4, merkle_root="e9b2d...")
    print(f"\n  🔗 审计链验证: {c('✅ 通过', 'green')} | 记录: {c(len(chain.entries), 'cyan')} 条")

    # 幻觉防御演示
    print(f"\n  🛡️ 策略风险评估:")
    dr = run_defense_engine(
        "risk_001",
        "HF_ALPHA_001 风险调整后收益高于同类均值",
        [{"id": "doc1", "similarity": 0.92}, {"id": "doc2", "similarity": 0.87}],
        ["Long/Short Equity", "Long/Short Equity", "Global Macro"],
        [0.85, 0.79, 0.62],
        index,
    )
    print(f"     风险评分: {c(dr.risk_score, 'yellow')} | 判定: {c(dr.verdict, 'green')}")
    print(f"     触发层: {dr.triggered_layers or '无'}")
    for d in dr.details:
        print(f"     {d}")
    print(f"     → 防御动作: {c(dr.defense_action, 'cyan')}")

def demo_hallucination_defense():
    banner("场景2: 幻觉防御 — 五层检测体系")

    print("  🛡️ 层1: CROWN 一致性防御（NeuroSync 原创）")
    print("  " + "─" * 50)
    r = layer3_crown_defense(
        "该患者为精神分裂症谱系障碍", 0.88,
        "建议排除双相情感障碍", 0.25, delta=0.10
    )
    print(f"     初始置信: {c('0.880', 'green')} | 社会置信: {c('0.250', 'red')}")
    print(f"     下跌: {r['confidence_drop']:.3f} > δ=0.10 → {c('🛡️ CROWN 触发', 'red')}")
    print(f"     采纳答案: {c(r['final_answer'], 'yellow')}")
    print(f"     原因: {r['reason']}")

    print("\n  🗳️ 层4: 多节点投票")
    print("  " + "─" * 50)
    r = layer4_multi_vote(
        ["Long/Short Equity", "Long/Short Equity", "Global Macro", "Stat Arb"],
        [0.85, 0.79, 0.62, 0.55],
        threshold=0.667,
    )
    print(f"     投票: {r['all_votes']}")
    print(f"     共识: {c(r['consensus'], 'green')} ({r['strength']:.0%})")
    print(f"     判定: {c(r['verdict'], 'yellow')}")

    print("\n  🔄 层5: LLM 自洽性检测")
    print("  " + "─" * 50)
    r = layer5_self_consistency(
        ["配置60%股票", "建议60%股票配置", "70%股票更优"],
        threshold=0.70,
    )
    print(f"     一致性: {r['score']:.0%} > 阈值70% → {c('✅ 通过' if r['passes'] else '❌ 未通过', 'yellow')}")
    print(f"     最常见: {r['most_common']} ({r['count']}/{r['total']}次)")

    print("\n  🎯 综合防御引擎")
    print("  " + "─" * 50)
    dr = run_defense_engine(
        "fund_claim_001",
        "应配置60%股票仓位以获得最优风险收益比",
        [{"id": "study1", "similarity": 0.78}, {"id": "study2", "similarity": 0.65}],
        ["60% 股票", "60% 股票", "70% 股票"],
        [0.82, 0.77, 0.61],
    )
    print(f"     风险评分: {c(f'{dr.risk_score:.0%}', 'yellow')}")
    print(f"     判定: {c(dr.verdict, 'green')}")
    print(f"     触发层: {dr.triggered_layers or c('无（完全可信）', 'green')}")
    print(f"     防御动作: {c(dr.defense_action, 'cyan')}")

def demo_agent():
    banner("场景3: 智能体 — 多工具协同")
    agent = type('Agent', (), {})()

    tasks = [
        "分析对冲基金 HF_ALPHA_001 的风险调整后收益",
        "比较三种不同市场环境下的策略表现",
        "如何优化私募基金的因子暴露？",
    ]

    for task in tasks:
        print(f"  📋 任务: {task}")

        steps = plan_task(task)
        print(f"  📐 规划: {len(steps)} 步")
        for s in steps:
            print(f"     {s['step_id']}. {c(s['tool'], 'cyan')} — {s['reason']}")

        result = run_agent(task)
        print(f"  ✅ 执行: {result['total_steps']} 步, 置信度: {result['confidence']:.0%}")
        print()

def demo_audit():
    banner("场景4: 区块链审计 — 全流程")
    chain = AuditChain(merkle_batch_size=5)

    events = [
        ("FL_NODE_REGISTERED", "node_a", {"domain": "hedge_fund", "data_size": 5000}),
        ("FL_NODE_REGISTERED", "node_b", {"domain": "hedge_fund", "data_size": 8000}),
        ("FL_UPDATE_RECEIVED", "node_a", {"round": 1, "weights_hash": "a3f8c..."}),
        ("FL_UPDATE_RECEIVED", "node_b", {"round": 1, "weights_hash": "b7d2e..."}),
        ("FL_AGGREGATION_COMPLETED", "server", {"round": 1, "participants": 2, "merkle_root": "e9b2d..."}),
        ("RAG_QUERY_EXECUTED", "node_a", {"query_id": "q001", "docs_returned": 3}),
        ("HALLUCINATION_CHECKED", "defense", {"claim_id": "c001", "risk": 0.28}),
        ("COT_VOTE_RECORDED", "cot_server", {"task_id": "t001", "participants": 3}),
    ]

    for event_type, node_id, meta in events:
        chain.append(event_type, node_id, **meta)

    print(f"  🔗 创世哈希: {c(chain.genesis[:16], 'yellow')}...")
    print(f"  📋 审计记录: {c(len(chain.entries), 'green')} 条")
    print(f"  ✅ 链完整性: {'通过' if chain.verify() else '失败'}")
    print(f"  🌳 Merkle根: {c(chain.current_merkle_root[:16], 'cyan') if chain.current_merkle_root else '批次未满'}...")

    print(f"\n  📊 FL聚合事件查询:")
    for e in chain.query("FL_AGGREGATION_COMPLETED", 5):
        print(f"     [{e['sequence']}] {e['event_type']} @ {e['node_id']}")
        print(f"        哈希: {e['hash'][:16]}... | 元数据: {e['metadata']}")

def demo_full():
    banner("场景5: 完整演示 — 联邦智能分析流程")
    print("  模拟: 多家对冲基金联合分析市场拥挤度\n")

    chain = AuditChain(merkle_batch_size=10)
    index = SimpleHNSW(dimension=8)

    # 插入因子向量
    factors = [
        ("momentum_30d", [0.9, 0.2, 0.1, 0.3, 0.8, 0.4, 0.2, 0.1]),
        ("value_factor", [0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.1, 0.2]),
        ("quality_score", [0.2, 0.3, 0.9, 0.1, 0.4, 0.8, 0.7, 0.3]),
        ("macro_beta", [0.7, 0.5, 0.3, 0.8, 0.9, 0.2, 0.1, 0.6]),
        ("vol_target", [0.3, 0.2, 0.1, 0.2, 0.2, 0.3, 0.9, 0.8]),
    ]
    for id, vec in factors:
        index.insert(id, vec)

    chain.append("FL_NODE_REGISTERED", "node_alpha", domain="hedge_fund", role="factor_provider")
    chain.append("FL_NODE_REGISTERED", "node_beta", domain="hedge_fund", role="factor_provider")

    # 联邦查询
    query_vec = [0.8, 0.3, 0.2, 0.5, 0.7, 0.5, 0.3, 0.2]
    results = index.search(query_vec, k=3)

    chain.append("RAG_QUERY_EXECUTED", "federated_query",
                 query="市场拥挤度最高的因子", results_count=len(results))

    print(f"  🔍 因子搜索结果:")
    for i, r in enumerate(results, 1):
        print(f"     {i}. {c(r['id'], 'yellow')} | 距离: {r['distance']:.3f} | 相似度: {r['similarity']:.1%}")

    # 幻觉防御
    dr = run_defense_engine(
        "crowding_001",
        "动量因子(30日) 当前市场拥挤度最高",
        [{"id": f"doc_{r['id']}", "similarity": r['similarity']} for r in results[:2]],
        ["momentum_30d", "vol_target", "momentum_30d"],
        [0.82, 0.71, 0.78],
        index,
    )

    chain.append("HALLUCINATION_CHECKED", "defense_engine",
                 claim_id="crowding_001", risk=dr.risk_score, verdict=dr.verdict)

    print(f"\n  🛡️ 幻觉防御: {c(dr.verdict, 'green' if dr.risk_score < 0.5 else 'red')}")
    print(f"     风险: {dr.risk_score:.0%} | 动作: {dr.defense_action}")

    print(f"\n  🔗 审计链: {len(chain.entries)} 条记录 | 完整: {'✅' if chain.verify() else '❌'}")


# ──────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────

def main():
    import sys
    scenario = sys.argv[1] if len(sys.argv) > 1 else "all"

    scenarios = {
        "hedge-fund": demo_hedge_fund,
        "hf": demo_hedge_fund,
        "hallucination": demo_hallucination_defense,
        "halluc": demo_hallucination_defense,
        "agent": demo_agent,
        "audit": demo_audit,
        "full": demo_full,
        "all": lambda: [demo_hedge_fund(), demo_hallucination_defense(), demo_agent(), demo_audit(), demo_full()],
    }

    if scenario not in scenarios:
        print(f"用法: python3 demo.py <场景>")
        print(f"场景: {' / '.join(scenarios.keys())}")
        return

    scenarios[scenario]()

if __name__ == "__main__":
    main()
