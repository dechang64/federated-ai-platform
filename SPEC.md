# Federated AI Platform — 联邦智能计算平台

## 1. Concept & Vision

以**自研 Rust HNSW 向量数据库**为核心引擎，构建隐私保护的联邦学习/联邦RAG/联邦CoT 智能体平台。

核心理念：**数据不动，模型动；数据不动，知识动；数据不动，推理动。**

平台服务于：对冲基金（策略隐私）、医疗机构（病例隐私）、制造业（工艺隐私）、文化遗产（非遗技艺保护）等高隐私敏感场景。

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Orchestration Layer                 │
│         (任务分解 · 多智能体协作 · 规划 · 反思)              │
└────────────────────────────┬────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼─────┐        ┌────▼──────┐       ┌────▼─────┐
   │  联邦RAG  │        │  联邦CoT   │       │  联邦学习  │
   │ (检索增强) │        │ (思维链)   │       │ (建模)    │
   └────┬─────┘        └────┬──────┘       └────┬─────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
   ┌────────────────────────▼────────────────────────────────┐
   │           Hallucination Defense Layer                    │
   │  (自研向量库检索核验 · 一致性检测 · CROWN防御 · 置信度)  │
   └────────────────────────┬────────────────────────────────┘
                             │
   ┌────────────────────────▼────────────────────────────────┐
   │              自研 Rust HNSW 向量数据库                    │
   │     (HNSW索引 · 欧式距离 · Rust Pure · <1ms)            │
   └────────────────────────┬────────────────────────────────┘
                             │
   ┌────────────────────────▼────────────────────────────────┐
   │                 Blockchain Audit Layer                     │
   │         (SHA-256哈希链 · Merkle树 · 操作不可篡改)        │
   └──────────────────────────────────────────────────────────┘
```

## 3. 核心模块

### 3.1 自研向量数据库 (vector_db)

- **语言**: Rust（纯自研，不依赖 faiss/milvus）
- **索引算法**: HNSW (Hierarchical Navigable Small World)
- **距离度量**: 欧氏距离（L2）、余弦相似度
- **核心接口**:
  - `insert(id, vector)` → 插入向量
  - `search(query, k, ef)` → kNN 搜索，ef 决定精度/速度权衡
  - `batch_insert(vectors)` → 批量插入
  - `update(id, vector)` → 更新向量
  - `delete(id)` → 删除向量
- **持久化**: SQLite + 内存索引双写
- **性能目标**: 100万向量，k=10，QPS > 1000，延迟 < 5ms

### 3.2 联邦学习 (fed_learn)

- **算法**: FedAvg (McMahan et al. 2017)
- **扩展**: Task-Aware Aggregation（基于任务表现加权）
- **隐私**: 本地差分隐私（可选，添加高斯噪声）
- **通信**: gRPC 流式传输
- **审计**: 每次参数更新记录 SHA-256 哈希链
- **核心接口**:
  - `FederatedServer::receive_update(client_id, weights)` → 接收客户端更新
  - `FederatedServer::aggregate()` → FedAvg 聚合
  - `FederatedClient::local_train(data)` → 本地训练
  - `FederatedClient::get_gradient()` → 获取梯度用于传输

### 3.3 联邦 RAG (rag)

- **目标**: 各机构本地知识库检索，隐私数据不出域
- **架构**:
  - 本地向量库：各机构持有
  - 中心聚合：只聚合查询意图embedding，不传原始文本
  - 检索增强：本地 top-k 结果 → LLM 生成答案
- **幻觉防御**: 
  - 检索结果相关性评分 < 阈值 → 标记为不可信
  - 每个答案附"引用溯源"（retrieval citations）
- **核心接口**:
  - `FederatedRAG::query(question, top_k)` → 跨机构隐私检索
  - `FederatedRAG::index_document(doc_id, text, metadata)` → 本地建索引

### 3.4 联邦 CoT (cot)

- **目标**: 分布式思维链推理，保护推理中间状态
- **架构**:
  - 各节点本地生成推理片段（Thought）
  - 中心节点协调：选择最优推理路径
  - 选择性暴露：只共享最终结论，中间步骤可选加密
- **幻觉防御**:
  - 多个节点独立推理 → 一致性投票
  - 偏离共识的推理路径 → 标记 + 警告
- **核心接口**:
  - `FederatedCoT::think(task, context)` → 分布式推理
  - `FederatedCoT::vote(thoughts)` → 多路径一致性投票

### 3.5 智能体平台 (agent)

- **架构**: ReAct (Reason + Act) 模式
- **组件**:
  - **任务规划器**：将复杂任务分解为子任务链
  - **工具调用器**：调度联邦RAG、联邦CoT、联邦建模
  - **记忆模块**：短期记忆（向量库）+ 长期记忆（文件）
  - **反思器**：检测执行偏差，自我修正
- **核心接口**:
  - `Agent::run(task)` → 执行任务
  - `Agent::plan(task)` → 规划分解
  - `Agent::reflect(result)` → 反思改进

### 3.6 幻觉检测与防御 (hallucination)

这是平台的核心护栏，参考 NeuroSync 的 CROWN 防御并扩展：

#### 防御层 1：检索一致性检测（Retrieval Consistency Check）
```
答案陈述 → 向量化 → 向量库检索 → 找到支撑文档？
  ├── 找到且相关性 > 0.85 → ✅ 可信
  ├── 找到但相关性 0.5~0.85 → ⚠️ 弱支撑，需标注
  └── 未找到或相关性 < 0.5 → 🔴 疑似幻觉
```

#### 防御层 2：自研向量库事实核验（Vector Fact-Check）
```
每个答案关键断言 → 向量化 → 在向量知识库中精确匹配
  ├── 匹配到 + 高相似度 → 引用支撑
  └── 匹配不到 → 标记为"无法证实"
```

#### 防御层 3：CROWN 一致性防御（参考 NeuroSync）
```
初始推理答案 → 参考外部知识 → 推理答案2
    ↓
置信度下跌幅度 > δ → 拒绝修改答案
```

#### 防御层 4：多节点一致性投票（Multi-Node Vote）
```
联邦CoT 各节点独立推理 → 各输出结论 → 投票
  ├── 共识结论（>2/3节点同意）→ 高可信 ✅
  ├── 分歧结论 → 标记不确定性 🔶
  └── 完全矛盾 → 触发人工复核 🔴
```

#### 防御层 5：LLM 自洽性检测（Self-Consistency）
```
同一问题，不同温度/推理路径 → 多次采样
    ↓
结论一致性 < 阈值 → 标记为不稳定
```

### 3.7 区块链审计 (audit)

- **结构**: SHA-256 哈希链（每条记录指向前一条的哈希）
- **可选升级**: Merkle 树（支持批量验证）
- **记录内容**:
  - 联邦学习：客户端注册、权重更新、聚合结果
  - 联邦 RAG：检索查询、本地文档访问、答案生成
  - 联邦 CoT：推理路径投票、结论采纳
  - 幻觉防御：检测触发、防御决策
- **可查询性**: HTTP API 查询任意时间点的审计快照

### 3.8 API 层 (api)

- **gRPC**: 联邦节点间通信（Rust 原生 tonic）
- **REST**: 外部调用（Python FastAPI / Rust Axum）
- **WebSocket**: 流式推理结果推送

## 4. 技术栈

| 层级 | 技术 |
|------|------|
| 核心引擎 | Rust（向量库、gRPC、区块链审计） |
| AI 建模 | Python（PyTorch FedAvg） |
| 向量索引 | 自研 Rust HNSW（复用现有项目） |
| 联邦通信 | gRPC（tonic） |
| 前端 | Streamlit（原型）|
| 部署 | Docker Compose（单机器多节点演示）|

## 5. 复用现有项目

```
自研 HNSW      ← 复用 FundFL/organoid-fl/embroidery-agent 的 hnsw_index.rs
区块链审计     ← 复用 organoid-fl 的 blockchain.rs（升级为 Merkle 树）
FedAvg         ← 复用 PAI 的 federated_learning.py（Rust 重写）
Portfolio Opt  ← 复用 PAI 的 portfolio_optimizer.py
Prototype Bank ← 复用 reading-fl 的 prototype_bank.py
CROWN 防御    ← 复用 NeuroSync 的 CROWN 实现
```

## 6. 开发阶段

- **Phase 1**: 核心框架 + 自研向量库 + 基本 API
- **Phase 2**: 联邦学习模块 + 审计链
- **Phase 3**: 联邦 RAG 模块
- **Phase 4**: 联邦 CoT 模块
- **Phase 5**: Agent 编排层
- **Phase 6**: 幻觉防御完整集成
- **Phase 7**: 性能优化 + 压力测试
