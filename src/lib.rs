//! 联邦智能计算平台 - lib.rs
//!
//! 以自研 Rust HNSW 向量数据库为核心：
//! - 联邦学习（FedAvg + 差分隐私）
//! - 联邦 RAG（隐私保护检索增强生成）
//! - 联邦 CoT（分布式思维链推理）
//! - 智能体平台（ReAct 任务编排）
//! - 幻觉防御（五层检测体系）

pub mod vector_db;
pub mod hallucination;
pub mod audit;
pub mod agent;

pub use vector_db::*;
pub use hallucination::*;
pub use audit::*;
pub use agent::*;
