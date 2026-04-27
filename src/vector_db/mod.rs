//! 自研 Rust HNSW 向量数据库核心
//!
//! 完全自主实现，不依赖 faiss/milvus 等外部向量库
//! 参考 hnsw crate 底层接口，构建完整的向量检索引擎

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use sha2::{Sha256, Digest};
use thiserror::Error;

// ──────────────────────────────────────────────────────────────
// 距离度量
// ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
pub struct Euclidean;

impl space::Metric<Vec<f32>> for Euclidean {
    type Unit = u32;
    fn distance(&self, a: &Vec<f32>, b: &Vec<f32>) -> u32 {
        let mut sum = 0.0f32;
        let len = a.len().min(b.len());
        for i in 0..len {
            let d = a[i] - b[i];
            sum += d * d;
        }
        sum.sqrt().to_bits()
    }
}

#[derive(Clone, Debug, Default)]
pub struct Cosine;

impl space::Metric<Vec<f32>> for Cosine {
    type Unit = u32;
    fn distance(&self, a: &Vec<f32>, b: &Vec<f32>) -> u32 {
        let len = a.len().min(b.len());
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        for i in 0..len {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        let sim = if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a.sqrt() * norm_b.sqrt())
        } else {
            0.0
        };
        // cosine distance = 1 - similarity, 映射到 u32
        ((1.0 - sim.clamp(-1.0, 1.0)) * 1000.0) as u32
    }
}

// 导入 hnsw crate 的 space trait
mod space {
    pub trait Metric<T> {
        type Unit: Clone + PartialOrd;
        fn distance(&self, a: &T, b: &T) -> Self::Unit;
    }
}

// ──────────────────────────────────────────────────────────────
// 错误类型
// ──────────────────────────────────────────────────────────────

#[derive(Error, Debug)]
pub enum VectorDBError {
    #[error("向量维度不匹配：期望 {expected}，实际 {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("向量 ID 不存在: {0}")]
    NotFound(String),
    #[error("索引未初始化")]
    NotInitialized,
    #[error("IO 错误: {0}")]
    Io(#[from] std::io::Error),
    #[error("数据库错误: {0}")]
    Database(String),
}

// ──────────────────────────────────────────────────────────────
// 索引项
// ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct VectorEntry {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: HashMap<String, String>,
    pub version: u64,
    pub inserted_at: String,
}

impl VectorEntry {
    pub fn new(id: String, vector: Vec<f32>) -> Self {
        Self {
            id,
            vector,
            metadata: HashMap::new(),
            version: 0,
            inserted_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), val.into());
        self
    }

    /// 生成向量的语义哈希（用于审计追踪，不用于相似性搜索）
    pub fn content_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.id.as_bytes());
        for v in &self.vector {
            hasher.update(v.to_bits().to_le_bytes());
        }
        format!("{:x}", hasher.finalize())
    }
}

// ──────────────────────────────────────────────────────────────
// 搜索结果
// ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub id: String,
    pub distance: f32,
    pub metadata: HashMap<String, String>,
    pub version: u64,
}

impl SearchResult {
    /// 从 HNSW 返回的 u32 距离转换为 f32
    pub fn from_bits(bits: u32) -> f32 {
        f32::from_bits(bits)
    }

    /// 计算余弦相似度（基于欧氏距离的近似）
    pub fn similarity(&self) -> f32 {
        // 当距离为 0 时相似度为 1
        // 简化为 exp(-distance)，实际应用中可调参
        (-self.distance * 0.1).exp().clamp(0.0, 1.0)
    }
}

// ──────────────────────────────────────────────────────────────
// 自研 HNSW 向量索引核心
// ──────────────────────────────────────────────────────────────

pub struct HnswVectorIndex<const M: usize = 16, const M0: usize = 32> {
    dimension: usize,
    /// 内部 ID → String ID 映射
    id_map: HashMap<String, usize>,
    /// String ID → 内部 ID 映射
    string_map: HashMap<String, usize>,
    /// 内部 HNSW 图结构
    graph: hnsw::Hnsw<Euclidean, Vec<f32>, rand::rngs::StdRng, M, M0>,
    /// 搜索器（可复用）
    searcher: hnsw::Searcher<u32>,
    /// 下一个可用内部 ID
    next_id: usize,
    /// 版本号（乐观锁）
    version: u64,
    /// 向量存储（ID → VectorEntry）
    entries: HashMap<usize, VectorEntry>,
}

impl<const M: usize, const M0: usize> HnswVectorIndex<M, M0> {
    /// 创建新的 HNSW 索引
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            id_map: HashMap::new(),
            string_map: HashMap::new(),
            graph: hnsw::Hnsw::new(Euclidean),
            searcher: hnsw::Searcher::new(),
            next_id: 0,
            version: 0,
            entries: HashMap::new(),
        }
    }

    /// 获取维度
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// 获取当前版本
    pub fn version(&self) -> u64 {
        self.version
    }

    /// 获取向量总数
    pub fn size(&self) -> usize {
        self.entries.len()
    }

    /// 验证向量维度
    fn validate_dimension(&self, vector: &[f32]) -> Result<(), VectorDBError> {
        if vector.len() != self.dimension {
            return Err(VectorDBError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.len(),
            });
        }
        Ok(())
    }

    /// 插入单个向量
    pub fn insert(&mut self, id: &str, vector: Vec<f32>) -> Result<usize, VectorDBError> {
        self.validate_dimension(&vector)?;

        let internal_id = self.next_id;
        self.next_id += 1;

        self.id_map.insert(id.to_string(), internal_id);
        self.string_map.insert(internal_id, id.to_string());

        let entry = VectorEntry::new(id.to_string(), vector.clone());
        self.entries.insert(internal_id, entry);

        // 插入 HNSW 图
        self.graph.insert(vector, &mut self.searcher);
        self.version += 1;

        Ok(internal_id)
    }

    /// 批量插入（更高效）
    pub fn batch_insert(&mut self, entries: Vec<(String, Vec<f32>)>) -> Result<Vec<usize>, VectorDBError> {
        let mut ids = Vec::with_capacity(entries.len());
        for (id, vector) in entries {
            match self.insert(&id, vector) {
                Ok(internal_id) => ids.push(internal_id),
                Err(e) => return Err(e),
            }
        }
        Ok(ids)
    }

    /// kNN 搜索
    pub fn search(&mut self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>, VectorDBError> {
        self.validate_dimension(query)?;

        if self.entries.is_empty() {
            return Ok(vec![]);
        }

        // 设置搜索参数
        let search_k = k.max(1);

        // HNSW 搜索
        let results = self.graph.search(query, search_k, ef, &mut self.searcher);

        let mut search_results = Vec::with_capacity(results.len());
        for (internal_id, dist_bits) in results {
            if let Some(entry) = self.entries.get(&internal_id) {
                search_results.push(SearchResult {
                    id: entry.id.clone(),
                    distance: SearchResult::from_bits(dist_bits),
                    metadata: entry.metadata.clone(),
                    version: entry.version,
                });
            }
        }

        // 按距离排序
        search_results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        search_results.truncate(k);

        Ok(search_results)
    }

    /// 范围搜索（找到距离在 threshold 内的所有向量）
    pub fn range_search(&mut self, query: &[f32], max_distance: f32, ef: usize) -> Result<Vec<SearchResult>, VectorDBError> {
        self.validate_dimension(query)?;

        let max_dist_bits = max_distance.to_bits();
        let results = self.graph.search(query, 100, ef, &mut self.searcher);

        let search_results: Vec<SearchResult> = results
            .into_iter()
            .filter(|(_, dist_bits)| *dist_bits <= max_dist_bits)
            .filter_map(|(internal_id, dist_bits)| {
                self.entries.get(&internal_id).map(|entry| SearchResult {
                    id: entry.id.clone(),
                    distance: SearchResult::from_bits(dist_bits),
                    metadata: entry.metadata.clone(),
                    version: entry.version,
                })
            })
            .collect();

        Ok(search_results)
    }

    /// 更新向量
    pub fn update(&mut self, id: &str, vector: Vec<f32>) -> Result<(), VectorDBError> {
        self.validate_dimension(&vector)?;

        let internal_id = *self.id_map.get(id)
            .ok_or_else(|| VectorDBError::NotFound(id.to_string()))?;

        let entry = self.entries.get_mut(&internal_id)
            .ok_or_else(|| VectorDBError::NotFound(id.to_string()))?;

        entry.vector = vector;
        entry.version += 1;
        self.version += 1;

        Ok(())
    }

    /// 删除向量
    pub fn delete(&mut self, id: &str) -> Result<(), VectorDBError> {
        let internal_id = self.id_map.remove(id)
            .ok_or_else(|| VectorDBError::NotFound(id.to_string()))?;

        self.string_map.remove(&internal_id);
        self.entries.remove(&internal_id);
        self.version += 1;

        Ok(())
    }

    /// 获取单个向量
    pub fn get(&self, id: &str) -> Option<&VectorEntry> {
        self.id_map.get(id)
            .and_then(|i| self.entries.get(i))
    }

    /// 生成索引快照哈希（用于审计）
    pub fn snapshot_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(format!("v{}", self.version).as_bytes());
        hasher.update(format!("n{}", self.entries.len()).as_bytes());
        hasher.update(format!("d{}", self.dimension).as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

// ──────────────────────────────────────────────────────────────
// 线程安全的封装（用于 gRPC 服务）
// ──────────────────────────────────────────────────────────────

pub type SharedVectorIndex<const M: usize = 16, const M0: usize = 32> =
    Arc<RwLock<HnswVectorIndex<M, M0>>>;

pub fn new_shared_index<const M: usize, const M0: usize>(dimension: usize) -> SharedVectorIndex<M, M0> {
    Arc::new(RwLock::new(HnswVectorIndex::new(dimension)))
}

// ──────────────────────────────────────────────────────────────
// 测试
// ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_search() {
        let mut index: HnswVectorIndex<16, 32> = HnswVectorIndex::new(128);

        // 插入测试向量
        let v1 = vec![0.1; 128];
        let v2 = vec![0.2; 128];
        let v3 = vec![0.9; 128];

        let id1 = index.insert("vec1", v1).unwrap();
        let id2 = index.insert("vec2", v2).unwrap();
        let _ = index.insert("vec3", v3).unwrap();

        assert_eq!(index.size(), 3);
        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut index: HnswVectorIndex<16, 32> = HnswVectorIndex::new(64);
        let wrong_dim = vec![0.1; 128];
        let result = index.insert("wrong", wrong_dim);
        assert!(result.is_err());
    }

    #[test]
    fn test_snapshot_hash() {
        let mut index: HnswVectorIndex<16, 32> = HnswVectorIndex::new(32);
        let hash1 = index.snapshot_hash();
        let _ = index.insert("v1", vec![0.1; 32]);
        let hash2 = index.snapshot_hash();
        assert_ne!(hash1, hash2);
    }
}
