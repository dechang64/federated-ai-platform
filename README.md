# Federated AI Platform — 联邦智能计算平台

<div align="center">

**以自研 Rust HNSW 向量数据库为核心，隐私保护的联邦学习 · RAG · CoT · Agent · 幻觉防御**

[![Rust](https://img.shields.io/badge/Rust-1.70+-orange?logo=rust)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

</div>

---

## 🏛️ 平台架构

```
┌──────────────────────────────────────────────────────────────┐
│                    Agent Orchestration                       │
│          (ReAct 任务分解 · 多智能体协作 · 反思)               │
└──────────────────────────┬───────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼─────┐       ┌────▼──────┐     ┌────▼─────┐
   │  联邦RAG  │       │  联邦CoT   │     │  联邦学习  │
   │ 检索增强  │       │ 思维链推理  │     │ FedAvg   │
   └────┬─────┘       └────┬──────┘     └────┬─────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
   ┌───────────────────────▼───────────────────────────────┐
   │              Hallucination Defense                    │
   │  (检索一致性 · 向量库核验 · CROWN · 多节点投票 · 自洽) │
   └───────────────────────┬───────────────────────────────┘
                           │
   ┌───────────────────────▼───────────────────────────────┐
   │           自研 Rust HNSW 向量数据库                   │
   │       (纯Rust · <1ms · 不依赖faiss/milvus)            │
   └───────────────────────┬───────────────────────────────┘
                           │
   ┌───────────────────────▼───────────────────────────────┐
   │              Blockchain Audit (SHA-256)               │
   │           哈希链 · Merkle树 · 操作不可篡改             │
   └───────────────────────────────────────────────────────┘
```

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🦀 **自研向量库** | Rust HNSW，不依赖外部向量库，kNN < 1ms |
| 🤝 **联邦学习** | FedAvg + Task-Aware，gRPC 流式，差分隐私可选 |
| 📚 **联邦 RAG** | 数据不出域，embedding 聚合，保护隐私检索 |
| 🧠 **联邦 CoT** | 分布式多步推理，多路径一致性投票 |
| 🤖 **Agent 编排** | ReAct 模式，任务自动分解，多工具协同 |
| 🛡️ **幻觉防御** | 五层检测：检索核验 + CROWN + 多节点投票 + 自洽性 |
| 🔗 **区块链审计** | SHA-256 哈希链 + Merkle 树，操作可追溯 |
| 🔐 **隐私保护** | 联邦架构，数据不动模型动，满足 GDPR 等合规要求 |

## 🛡️ 五层幻觉防御体系

```
防御层 1 ──→ 检索一致性检测
              答案断言 → 向量化 → 向量库检索
                ├ 相似度 > 0.85 → ✅ 可信
                └ 相似度 < 0.50 → 🔴 疑似幻觉

防御层 2 ──→ 自研向量库事实核验
              关键术语 → 精确匹配 → 支撑引用

防御层 3 ──→ CROWN 一致性防御（NeuroSync 原创）
              独立推理 → 参考外部 → 置信度下跌 > δ → 拒绝

防御层 4 ──→ 多节点一致性投票
              联邦 CoT 多节点独立推理 → 共识投票

防御层 5 ──→ LLM 自洽性检测
              同一问题多次采样 → 答案一致性 < 阈值 → 标记
```

## 📁 项目结构

```
federated-ai-platform/
├── src/
│   ├── main.rs                 # 入口 + CLI + 演示场景
│   ├── lib.rs                  # 库导出
│   ├── vector_db/
│   │   └── mod.rs              # 自研 Rust HNSW 向量库
│   ├── hallucination/
│   │   └── mod.rs             # 五层幻觉防御引擎
│   ├── audit/
│   │   └── mod.rs             # SHA-256 哈希链 + Merkle 树
│   └── agent/
│       └── mod.rs             # ReAct 智能体编排
├── proto/
│   └── federated_ai.proto     # gRPC 接口定义
├── Cargo.toml
├── build.rs
└── SPEC.md                     # 完整技术规格书
```

## 🚀 快速开始

```bash
# 编译
cargo build --release

# 运行全部演示
cargo run -- --demo all

# 运行幻觉防御演示
cargo run -- --demo hallucination

# 运行对冲基金场景
cargo run -- --demo hedge-fund

# 交互式 CLI
cargo run -- --interactive
```

## 🔌 复用现有项目

本平台复用了以下项目的核心代码：

| 源项目 | 复用的技术 |
|--------|----------|
| FundFL | Rust HNSW 向量索引 (hnsw_index.rs) |
| organoid-fl | SHA-256 区块链审计 (blockchain.rs) |
| PAI | FedAvg 聚合器 + Portfolio Optimizer |
| NeuroSync | CROWN Conformity 防御 |
| reading-fl | Quality Prototype Bank |
| embroidery-agent | gRPC 服务框架 |

## 📊 目标应用场景

| 场景 | 痛点 | 本平台解决方案 |
|------|------|--------------|
| **对冲基金** | 策略持仓隐私 | 联邦因子分析，模型共享，持仓不外泄 |
| **医疗机构** | 病例数据隐私 | 联邦医学影像建模，数据不出院 |
| **制造业** | 工艺参数保密 | 联邦质量检测，工艺不外流 |
| **文化遗产** | 非遗技艺保护 | 联邦刺绣/壁画知识库，技艺不外传 |

## 📄 License

Apache-2.0
