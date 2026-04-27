//! 区块链审计层
//!
//! 采用 SHA-256 哈希链（不可篡改）+ 可选 Merkle 树（支持批量验证）
//! 记录所有联邦操作：学习更新、RAG 查询、CoT 投票、幻觉防御决策

use sha2::{Sha256, Digest};
use serde::{Serialize, Deserialize};
use thiserror::Error;
use std::collections::HashMap;

#[derive(Error, Debug)]
pub enum AuditError {
    #[error("链式断裂：期望哈希 {expected}，实际 {actual}")]
    ChainBroken { expected: String, actual: String },
    #[error("序列号不连续：期望 {expected}，实际 {actual}")]
    SequenceGap { expected: u64, actual: u64 },
    #[error("无效的事件类型: {0}")]
    InvalidEventType(String),
}

// ──────────────────────────────────────────────────────────────
// 事件类型
// ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum EventType {
    // 联邦学习
    FlNodeRegistered,
    FlUpdateReceived,
    FlAggregationCompleted,
    // 联邦 RAG
    RagDocumentIndexed,
    RagQueryExecuted,
    RagDefenseTriggered,
    // 联邦 CoT
    CotThoughtGenerated,
    CotVoteRecorded,
    CotConsensusReached,
    // 幻觉防御
    HallucinationChecked,
    HallucinationDetected,
    CrownDefenseTriggered,
    // 向量库
    VectorInserted,
    VectorDeleted,
    VectorUpdated,
}

impl EventType {
    pub fn as_str(&self) -> &'static str {
        match self {
            EventType::FlNodeRegistered => "FL_NODE_REGISTERED",
            EventType::FlUpdateReceived => "FL_UPDATE_RECEIVED",
            EventType::FlAggregationCompleted => "FL_AGGREGATION_COMPLETED",
            EventType::RagDocumentIndexed => "RAG_DOCUMENT_INDEXED",
            EventType::RagQueryExecuted => "RAG_QUERY_EXECUTED",
            EventType::RagDefenseTriggered => "RAG_DEFENSE_TRIGGERED",
            EventType::CotThoughtGenerated => "COT_THOUGHT_GENERATED",
            EventType::CotVoteRecorded => "COT_VOTE_RECORDED",
            EventType::CotConsensusReached => "COT_CONSENSUS_REACHED",
            EventType::HallucinationChecked => "HALLUCINATION_CHECKED",
            EventType::HallucinationDetected => "HALLUCINATION_DETECTED",
            EventType::CrownDefenseTriggered => "CROWN_DEFENSE_TRIGGERED",
            EventType::VectorInserted => "VECTOR_INSERTED",
            EventType::VectorDeleted => "VECTOR_DELETED",
            EventType::VectorUpdated => "VECTOR_UPDATED",
        }
    }
}

// ──────────────────────────────────────────────────────────────
// 审计记录
// ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditEntry {
    /// 序列号（递增）
    pub sequence: u64,
    /// 时间戳（ISO 8601）
    pub timestamp: String,
    /// 事件类型
    pub event_type: String,
    /// 节点 ID
    pub node_id: String,
    /// 事件内容哈希（SHA-256）
    pub payload_hash: String,
    /// 前一条记录的哈希
    pub prev_hash: String,
    /// 当前记录的哈希
    pub hash: String,
    /// Merkle 证明（可选，用于批量验证）
    pub merkle_proof: Option<MerkleProof>,
    /// 事件元数据（JSON 格式）
    pub metadata: HashMap<String, String>,
}

impl AuditEntry {
    /// 计算事件的 payload 哈希
    fn compute_payload_hash(event_type: &str, node_id: &str, metadata: &HashMap<String, String>) -> String {
        let mut hasher = Sha256::new();
        hasher.update(event_type.as_bytes());
        hasher.update(node_id.as_bytes());
        // 稳定的序列化顺序
        let mut sorted_keys: Vec<_> = metadata.keys().collect();
        sorted_keys.sort();
        for key in sorted_keys {
            hasher.update(key.as_bytes());
            hasher.update(metadata.get(key).unwrap().as_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    /// 计算整条记录的哈希（包含 prev_hash，形成链）
    fn compute_hash(seq: u64, ts: &str, event_type: &str, node_id: &str, payload_hash: &str, prev_hash: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(seq.to_le_bytes());
        hasher.update(ts.as_bytes());
        hasher.update(event_type.as_bytes());
        hasher.update(node_id.as_bytes());
        hasher.update(payload_hash.as_bytes());
        hasher.update(prev_hash.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// 创建新的审计条目
    pub fn new(
        sequence: u64,
        event_type: EventType,
        node_id: String,
        metadata: HashMap<String, String>,
        prev_hash: &str,
    ) -> Self {
        let timestamp = chrono::Utc::now().to_rfc3339();
        let event_str = event_type.as_str();
        let payload_hash = Self::compute_payload_hash(event_str, &node_id, &metadata);
        let hash = Self::compute_hash(sequence, &timestamp, event_str, &node_id, &payload_hash, prev_hash);

        Self {
            sequence,
            timestamp,
            event_type: event_str.to_string(),
            node_id,
            payload_hash,
            prev_hash: prev_hash.to_string(),
            hash,
            merkle_proof: None,
            metadata,
        }
    }

    /// 验证链的完整性（检查当前条目的 prev_hash 是否等于前一条的 hash）
    pub fn verify_chain(&self, prev_entry: &AuditEntry) -> Result<(), AuditError> {
        if prev_entry.sequence + 1 != self.sequence {
            return Err(AuditError::SequenceGap {
                expected: prev_entry.sequence + 1,
                actual: self.sequence,
            });
        }
        if prev_entry.hash != self.prev_hash {
            return Err(AuditError::ChainBroken {
                expected: prev_entry.hash.clone(),
                actual: self.prev_hash.clone(),
            });
        }
        Ok(())
    }

    /// 生成 Merkle 证明
    pub fn generate_merkle_proof(&self, sibling_hashes: Vec<String>) -> MerkleProof {
        MerkleProof {
            leaf_hash: self.hash.clone(),
            sibling_hashes,
            tree_height: 0, // 由 AuditChain 在生成时设置
        }
    }
}

// ──────────────────────────────────────────────────────────────
// Merkle 树（可选，用于批量验证）
// ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MerkleProof {
    pub leaf_hash: String,
    pub sibling_hashes: Vec<String>,
    pub tree_height: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MerkleTree {
    pub root: String,
    pub leaves: Vec<String>,
    pub tree: Vec<Vec<String>>,  // 每层的哈希列表
}

impl MerkleTree {
    /// 从叶子节点构建 Merkle 树
    pub fn build(leaves: Vec<String>) -> Self {
        if leaves.is_empty() {
            return Self {
                root: Sha256::digest("empty").to_string(),
                leaves: vec![],
                tree: vec![],
            };
        }

        let mut current_level = leaves.clone();
        let mut all_levels = vec![current_level.clone()];

        while current_level.len() > 1 {
            let mut next_level = Vec::new();
            for chunk in current_level.chunks(2) {
                let combined = if chunk.len() == 2 {
                    format!("{}{}", chunk[0], chunk[1])
                } else {
                    format!("{}{}", chunk[0], chunk[0]) // 奇数节点复制自己
                };
                let hash = format!("{:x}", Sha256::digest(&combined));
                next_level.push(hash);
            }
            current_level = next_level;
            all_levels.push(current_level.clone());
        }

        let root = current_level[0].clone();
        Self { root, leaves, tree: all_levels }
    }

    /// 生成指定叶子节点的 Merkle 证明
    pub fn prove(&self, leaf_index: usize) -> Option<MerkleProof> {
        if leaf_index >= self.leaves.len() {
            return None;
        }

        let mut sibling_hashes = Vec::new();
        let mut idx = leaf_index;

        for level in 0..self.tree.len() - 1 {
            let is_right = idx % 2 == 1;
            let level_nodes = &self.tree[level];

            if is_right && idx > 0 {
                sibling_hashes.push(level_nodes[idx - 1].clone());
            } else if !is_right && idx + 1 < level_nodes.len() {
                sibling_hashes.push(level_nodes[idx + 1].clone());
            } else {
                // 同级没有兄弟，复制自己
                sibling_hashes.push(level_nodes[idx].clone());
            }

            idx = idx / 2;
        }

        Some(MerkleProof {
            leaf_hash: self.leaves[leaf_index].clone(),
            sibling_hashes,
            tree_height: (self.tree.len() - 1) as u32,
        })
    }

    /// 验证 Merkle 证明
    pub fn verify(proof: &MerkleProof, root: &str) -> bool {
        let mut current = proof.leaf_hash.clone();

        for (i, sibling) in proof.sibling_hashes.iter().enumerate() {
            let combined = if i % 2 == 0 {
                format!("{}{}", current, sibling)
            } else {
                format!("{}{}", sibling, current)
            };
            current = format!("{:x}", Sha256::digest(&combined));
        }

        current == *root
    }
}

// ──────────────────────────────────────────────────────────────
// 审计链
// ──────────────────────────────────────────────────────────────

pub struct AuditChain {
    /// 创世块哈希（固定值）
    genesis_hash: String,
    /// 当前链（内存中）
    entries: Vec<AuditEntry>,
    /// 当前序列号
    next_sequence: u64,
    /// Merkle 树（每批审计打包）
    merkle_batch_size: usize,
    /// 当前批次的 Merkle 树
    current_merkle: Option<MerkleTree>,
    /// 当前批次叶子数
    batch_count: usize,
}

impl AuditChain {
    /// 创建新的审计链
    pub fn new(merkle_batch_size: usize) -> Self {
        // 创世块
        let genesis_hash = format!("{:x}", Sha256::digest(b"GENESIS_FEDERATED_AI_PLATFORM_v1"));
        Self {
            genesis_hash,
            entries: Vec::new(),
            next_sequence: 1,
            merkle_batch_size,
            current_merkle: None,
            batch_count: 0,
        }
    }

    /// 创世哈希
    pub fn genesis(&self) -> &str {
        &self.genesis_hash
    }

    /// 添加审计条目
    pub fn append(&mut self, event_type: EventType, node_id: String, metadata: HashMap<String, String>) -> &AuditEntry {
        let prev_hash = if let Some(last) = self.entries.last() {
            last.hash.clone()
        } else {
            self.genesis_hash.clone()
        };

        let entry = AuditEntry::new(self.next_sequence, event_type, node_id, metadata, &prev_hash);
        self.next_sequence += 1;

        // 更新 Merkle 树
        self.batch_count += 1;
        if self.batch_count == 1 {
            self.current_merkle = Some(MerkleTree::build(vec![entry.hash.clone()]));
        } else {
            // 追加到当前 Merkle 树
            if let Some(ref mut mt) = self.current_merkle {
                let mut new_leaves = mt.leaves.clone();
                new_leaves.push(entry.hash.clone());
                *mt = MerkleTree::build(new_leaves);
            }
        }

        // 满批次时重置
        if self.batch_count >= self.merkle_batch_size {
            self.batch_count = 0;
            self.current_merkle = None;
        }

        self.entries.push(entry);
        self.entries.last().unwrap()
    }

    /// 获取链长度
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 获取指定序列号的条目
    pub fn get(&self, sequence: u64) -> Option<&AuditEntry> {
        self.entries.iter().find(|e| e.sequence == sequence)
    }

    /// 获取最新条目
    pub fn latest(&self) -> Option<&AuditEntry> {
        self.entries.last()
    }

    /// 验证整条链的完整性
    pub fn verify(&self) -> Result<(), Vec<AuditError>> {
        let mut errors = Vec::new();
        let mut expected_seq = 1u64;
        let mut expected_prev = self.genesis_hash.clone();

        for entry in &self.entries {
            if entry.sequence != expected_seq {
                errors.push(AuditError::SequenceGap {
                    expected: expected_seq,
                    actual: entry.sequence,
                });
            }
            if entry.prev_hash != expected_prev {
                errors.push(AuditError::ChainBroken {
                    expected: expected_prev,
                    actual: entry.prev_hash.clone(),
                });
            }
            expected_seq += 1;
            expected_prev = entry.hash.clone();
        }

        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }

    /// 按事件类型查询
    pub fn query(&self, event_type: Option<EventType>, node_id: Option<&str>, limit: usize) -> Vec<&AuditEntry> {
        self.entries.iter()
            .filter(|e| {
                let type_match = event_type.as_ref()
                    .map(|t| e.event_type == t.as_str())
                    .unwrap_or(true);
                let node_match = node_id
                    .map(|n| e.node_id == n)
                    .unwrap_or(true);
                type_match && node_match
            })
            .rev()  // 最新在前
            .take(limit)
            .collect()
    }

    /// 导出链（用于持久化）
    pub fn export(&self) -> ChainExport {
        ChainExport {
            genesis_hash: self.genesis_hash.clone(),
            entries: self.entries.clone(),
            next_sequence: self.next_sequence,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChainExport {
    pub genesis_hash: String,
    pub entries: Vec<AuditEntry>,
    pub next_sequence: u64,
}

// 便捷宏：快速记录审计事件
#[macro_export]
macro_rules! audit {
    ($chain:expr, $event:expr, $node:expr; $($key:ident = $value:expr),* $(,)?) => {{
        use std::collections::HashMap;
        let mut meta = HashMap::new();
        $( meta.insert(stringify!($key).to_string(), $value.to_string()); )*
        $chain.append($event, $node.to_string(), meta)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_append_and_verify() {
        let mut chain = AuditChain::new(10);

        let e1 = audit!(chain, EventType::FlNodeRegistered, "node1"; data_size = 1000);
        let e2 = audit!(chain, EventType::FlUpdateReceived, "node1"; round = 1, samples = 500);
        let e3 = audit!(chain, EventType::FlAggregationCompleted, "server"; round = 1, participants = 3);

        assert_eq!(e1.sequence, 1);
        assert_eq!(e2.sequence, 2);
        assert_eq!(e3.sequence, 3);
        assert_eq!(e2.prev_hash, e1.hash);
        assert_eq!(e3.prev_hash, e2.hash);

        assert!(chain.verify().is_ok());
    }

    #[test]
    fn test_merkle_tree() {
        let leaves = vec![
            "leaf1".to_string(),
            "leaf2".to_string(),
            "leaf3".to_string(),
        ];
        let tree = MerkleTree::build(leaves.clone());
        let proof = tree.prove(0).unwrap();
        assert!(MerkleTree::verify(&proof, &tree.root));
    }
}
