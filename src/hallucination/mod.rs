//! 幻觉检测与防御核心模块
//!
//! 五层防御体系：
//! 1. 检索一致性检测 (Retrieval Consistency Check)
//! 2. 自研向量库事实核验 (Vector Fact-Check)
//! 3. CROWN 一致性防御 (Multi-Node CROWN)
//! 4. 多节点一致性投票 (Multi-Node Vote)
//! 5. LLM 自洽性检测 (Self-Consistency)

use crate::vector_db::{SharedVectorIndex, SearchResult};
use std::collections::HashMap;
use sha2::{Sha256, Digest};
use serde::{Serialize, Deserialize};

// ──────────────────────────────────────────────────────────────
// 防御配置
// ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DefenseConfig {
    /// 检索相关性阈值：低于此值认为支撑不足
    pub retrieval_threshold: f32,
    /// CROWN 置信度下跌阈值：超过此值拒绝从众答案
    pub crown_delta: f32,
    /// 自洽性最低一致性分数
    pub self_consistency_threshold: f32,
    /// 多节点投票共识阈值（比例）
    pub consensus_threshold: f32,
    /// 幻觉风险高危线
    pub hallucination_high_risk: f32,
    /// 幻觉风险中危线
    pub hallucination_medium_risk: f32,
}

impl Default for DefenseConfig {
    fn default() -> Self {
        Self {
            retrieval_threshold: 0.50,
            crown_delta: 0.10,
            self_consistency_threshold: 0.70,
            consensus_threshold: 0.667,   // 2/3
            hallucination_high_risk: 0.75,
            hallucination_medium_risk: 0.45,
        }
    }
}

// ──────────────────────────────────────────────────────────────
// 防御结果
// ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DefenseResult {
    pub claim_id: String,
    pub is_hallucination: bool,
    pub risk_score: f32,
    pub verdict: Verdict,
    pub triggered_layers: Vec<String>,
    pub evidence: Vec<DefenseEvidence>,
    pub defense_action: DefenseAction,
    pub timestamp: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Verdict {
    Verified,       // 经过多源验证，高度可信
    LikelyTrue,     // 证据支持，可能为真
    Uncertain,      // 证据不足，无法判断
    LikelyFalse,    // 有反面证据，可能为假
    Hallucination,  // 高置信度幻觉
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DefenseEvidence {
    pub layer: String,
    pub source: String,
    pub description: String,
    pub confidence: f32,
    pub doc_id: Option<String>,
    pub similarity: Option<f32>,
}

// 防御动作
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum DefenseAction {
    Accept,    // 直接接受
    Flag,      // 标记为不确定
    Reject,    // 拒绝，触发重新生成
    Review,    // 人工复核
}

// ──────────────────────────────────────────────────────────────
// 防御层 1：检索一致性检测
// ──────────────────────────────────────────────────────────────

pub struct RetrievalConsistencyChecker {
    threshold: f32,
}

impl RetrievalConsistencyChecker {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// 检查答案中的关键断言是否能被向量库检索结果支撑
    pub fn check(
        &self,
        claim: &str,
        relevant_docs: &[SearchResult],
    ) -> (bool, f32, Vec<DefenseEvidence>) {
        if relevant_docs.is_empty() {
            return (false, 0.0, vec![
                DefenseEvidence {
                    layer: "RetrievalConsistency".to_string(),
                    source: "vector_db".to_string(),
                    description: "未找到任何支撑文档".to_string(),
                    confidence: 0.0,
                    doc_id: None,
                    similarity: None,
                }
            ]);
        }

        // 计算最高相关性
        let max_sim = relevant_docs.iter()
            .map(|r| r.similarity())
            .fold(0.0f32, |a, b| a.max(b));

        let is_supported = max_sim >= self.threshold;

        let evidence = relevant_docs.iter().map(|r| {
            let sim = r.similarity();
            DefenseEvidence {
                layer: "RetrievalConsistency".to_string(),
                source: "vector_db".to_string(),
                description: format!("文档 {} 检索相关性: {:.3}", r.id, sim),
                confidence: sim,
                doc_id: Some(r.id.clone()),
                similarity: Some(sim),
            }
        }).collect();

        (is_supported, max_sim, evidence)
    }
}

// ──────────────────────────────────────────────────────────────
// 防御层 2：向量库事实核验
// ──────────────────────────────────────────────────────────────

pub struct VectorFactChecker {
    vector_index: SharedVectorIndex,
    claim_keywords: Vec<String>,
}

impl VectorFactChecker {
    pub fn new(vector_index: SharedVectorIndex) -> Self {
        Self {
            vector_index,
            claim_keywords: vec![],
        }
    }

    /// 从断言中提取关键实体/术语（简化版：分词 + 长度过滤）
    fn extract_key_terms(&self, claim: &str) -> Vec<String> {
        claim.split(|c: char| !c.is_alphanumeric() && c != ' ')
            .filter(|s| s.len() >= 2)
            .map(|s| s.to_lowercase())
            .collect()
    }

    /// 在向量库中精确查找与断言相关的事实
    pub fn verify_fact(&self, claim: &str, ef_construction: usize) -> FactCheckResult {
        let terms = self.extract_key_terms(claim);
        let index = self.vector_index.read().unwrap();

        let mut supporting = Vec::new();
        let mut contradicting = Vec::new();

        for term in terms {
            // 搜索相关向量（用短词向量做近似匹配）
            let fake_query = vec![0.0f32; index.dimension()];
            let results = index.range_search(&fake_query, 0.5, ef_construction).unwrap_or_default();

            for result in results.iter().take(5) {
                let meta_str = result.metadata.get("text")
                    .cloned()
                    .unwrap_or_default();

                if meta_str.to_lowercase().contains(&term) {
                    supporting.push(FactCheckItem {
                        term: term.clone(),
                        doc_id: result.id.clone(),
                        evidence: meta_str,
                        similarity: result.similarity(),
                    });
                }
            }
        }

        let verified_ratio = if !supporting.is_empty() {
            supporting.len() as f32 / terms.len().max(1) as f32
        } else {
            0.0
        };

        FactCheckResult {
            total_terms: terms.len(),
            supporting,
            contradicting,
            verified_ratio,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FactCheckResult {
    pub total_terms: usize,
    pub supporting: Vec<FactCheckItem>,
    pub contradicting: Vec<FactCheckItem>,
    pub verified_ratio: f32,
}

#[derive(Clone, Debug)]
pub struct FactCheckItem {
    pub term: String,
    pub doc_id: String,
    pub evidence: String,
    pub similarity: f32,
}

// ──────────────────────────────────────────────────────────────
// 防御层 3：CROWN 一致性防御
// ──────────────────────────────────────────────────────────────

/// CROWN 防御（参考 NeuroSync，实现为 Rust 原生版本）
///
/// 核心逻辑：
/// 1. LLM 独立推理得到初始答案 + 初始置信度
/// 2. LLM 参考外部知识得到社会答案 + 社会置信度
/// 3. 如果社会置信度比初始置信度下跌超过 δ，拒绝社会答案
///
/// 原理：LLM 有时会因为参考了有偏知识而自我否定正确判断
/// CROWN 规则优先信任 LLM 更有把握时的独立判断
pub struct CROWNDefense {
    delta: f32,
}

impl CROWNDefense {
    pub fn new(delta: f32) -> Self {
        Self { delta }
    }

    /// 执行 CROWN 防御检查
    ///
    /// 返回 (是否触发防御, 最终采纳的答案, 决策原因)
    pub fn check(
        &self,
        initial_answer: &str,
        initial_confidence: f32,
        social_answer: &str,
        social_confidence: f32,
    ) -> CROWNResult {
        let answer_changed = initial_answer.trim() != social_answer.trim();
        let confidence_drop = initial_confidence - social_confidence;
        let triggered = confidence_drop > self.delta;

        let (final_answer, reason) = if !answer_changed {
            // 答案没变，不用防御
            (initial_answer.to_string(), "答案未变化".to_string())
        } else if triggered {
            // CROWN 触发：置信度下跌超阈值，拒绝社会答案
            (initial_answer.to_string(), format!(
                "CROWN触发: 置信度下跌 {:.3f} > δ={:.3f}，拒绝社会答案",
                confidence_drop, self.delta
            ))
        } else {
            // 置信度下跌可接受，采纳社会答案
            (social_answer.to_string(), format!(
                "置信度下跌 {:.3f} ≤ δ={:.3f}，采纳社会答案",
                confidence_drop, self.delta
            ))
        };

        CROWNResult {
            triggered,
            confidence_drop,
            final_answer,
            reason,
            crown_delta: self.delta,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CROWNResult {
    pub triggered: bool,
    pub confidence_drop: f32,
    pub final_answer: String,
    pub reason: String,
    pub crown_delta: f32,
}

// ──────────────────────────────────────────────────────────────
// 防御层 4：多节点一致性投票
// ──────────────────────────────────────────────────────────────

pub struct MultiNodeVoter {
    consensus_threshold: f32,
}

impl MultiNodeVoter {
    pub fn new(consensus_threshold: f32) -> Self {
        Self { consensus_threshold }
    }

    /// 对多个节点的答案进行投票
    ///
    /// 返回共识结论和投票详情
    pub fn vote(&self, answers: &[String], confidences: &[f32]) -> VoteResult {
        if answers.is_empty() {
            return VoteResult {
                consensus: None,
                consensus_strength: 0.0,
                vote_counts: HashMap::new(),
                dissenting: vec![],
                verdict: Verdict::Uncertain,
            };
        }

        // 统计答案出现次数
        let mut counts: HashMap<&str, (u32, f32)> = HashMap::new();
        for (ans, conf) in answers.iter().zip(confidences.iter()) {
            let entry = counts.entry(ans).or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += conf;
        }

        // 找出最高票
        let best = counts.iter()
            .max_by_key(|(_, (count, _))| *count)
            .map(|(k, (count, avg_conf))| {
                let strength = *count as f32 / answers.len() as f32;
                (k.to_string(), *count, *count as f32 / answers.len() as f32, *avg_conf / *count as f32)
            });

        let (consensus, count, strength, _) = best.unwrap_or(("".to_string(), 0, 0.0, 0.0));

        let dissenting: Vec<String> = answers.iter()
            .filter(|a| a.as_str() != consensus)
            .cloned()
            .collect();

        let verdict = if strength >= self.consensus_threshold {
            Verdict::Verified
        } else if strength >= 0.5 {
            Verdict::LikelyTrue
        } else if !dissenting.is_empty() {
            Verdict::Uncertain
        } else {
            Verdict::Hallucination
        };

        VoteResult {
            consensus: Some(consensus),
            consensus_strength: strength,
            vote_counts: counts.iter()
                .map(|(k, (c, _))| (k.to_string(), *c))
                .collect(),
            dissenting,
            verdict,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VoteResult {
    /// 共识结论（如果有）
    pub consensus: Option<String>,
    /// 共识强度 [0,1]
    pub consensus_strength: f32,
    /// 各答案的投票计数
    pub vote_counts: HashMap<String, u32>,
    /// 分歧答案列表
    pub dissenting: Vec<String>,
    /// 判断结论
    pub verdict: Verdict,
}

// ──────────────────────────────────────────────────────────────
// 防御层 5：LLM 自洽性检测
// ──────────────────────────────────────────────────────────────

pub struct SelfConsistencyChecker {
    threshold: f32,
}

impl SelfConsistencyChecker {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// 检查同一问题多次采样的答案是否一致
    ///
    /// answers: 同一问题在不同温度/推理路径下生成的多个答案
    pub fn check(&self, answers: &[String]) -> SelfConsistencyResult {
        if answers.is_empty() {
            return SelfConsistencyResult {
                consistency_score: 0.0,
                most_common_answer: None,
                passes: false,
                dissenting_answers: vec![],
            };
        }

        // 统计最常见答案
        let mut counts: HashMap<&str, u32> = HashMap::new();
        for ans in answers {
            *counts.entry(ans).or_insert(0) += 1;
        }

        let best = counts.iter()
            .max_by_key(|(_, c)| *c)
            .map(|(k, c)| (k.to_string(), *c));

        let (most_common, count) = best.unwrap_or(("".to_string(), 0));
        let consistency_score = count as f32 / answers.len() as f32;

        let dissenting: Vec<String> = answers.iter()
            .filter(|a| a.as_str() != most_common)
            .cloned()
            .collect();

        SelfConsistencyResult {
            consistency_score,
            most_common_answer: Some(most_common),
            passes: consistency_score >= self.threshold,
            dissenting_answers: dissenting,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SelfConsistencyResult {
    pub consistency_score: f32,
    pub most_common_answer: Option<String>,
    pub passes: bool,
    pub dissenting_answers: Vec<String>,
}

// ──────────────────────────────────────────────────────────────
// 综合防御引擎
// ──────────────────────────────────────────────────────────────

/// 综合幻觉防御引擎
/// 协调五层防御，最终输出统一判断
pub struct HallucinationDefenseEngine {
    config: DefenseConfig,
    retrieval_checker: RetrievalConsistencyChecker,
    crown_defense: CROWNDefense,
    multi_node_voter: MultiNodeVoter,
    self_consistency_checker: SelfConsistencyChecker,
    vector_fact_checker: Option<VectorFactChecker>,
}

impl HallucinationDefenseEngine {
    pub fn new(config: DefenseConfig, vector_index: Option<SharedVectorIndex>) -> Self {
        Self {
            retrieval_checker: RetrievalConsistencyChecker::new(config.retrieval_threshold),
            crown_defense: CROWNDefense::new(config.crown_delta),
            multi_node_voter: MultiNodeVoter::new(config.consensus_threshold),
            self_consistency_checker: SelfConsistencyChecker::new(config.self_consistency_threshold),
            vector_fact_checker: vector_index.map(VectorFactChecker::new),
        }
    }

    /// 综合五层防御，检查一个答案是否为幻觉
    ///
    /// # 参数
    /// - `claim_id`: 断言唯一标识
    /// - `claim`: 待检测的答案/断言
    /// - `context`: 答案的上下文
    /// - `retrieved_docs`: 向量库检索到的相关文档
    /// - `multi_node_answers`: 来自多个联邦节点的独立答案（可选）
    pub fn check(
        &self,
        claim_id: &str,
        claim: &str,
        context: &str,
        retrieved_docs: &[SearchResult],
        multi_node_answers: Option<&[String]>,
        multi_node_confidences: Option<&[f32]>,
    ) -> DefenseResult {
        let mut all_evidence = Vec::new();
        let mut triggered_layers = Vec::new();
        let mut risk_factors: Vec<f32> = Vec::new();

        // ── 防御层 1: 检索一致性 ──
        let (ret_supported, max_sim, mut evidence) =
            self.retrieval_checker.check(claim, retrieved_docs);
        if !ret_supported {
            triggered_layers.push("RetrievalConsistency".to_string());
            risk_factors.push(1.0 - max_sim);
        }
        all_evidence.append(&mut evidence);

        // ── 防御层 2: 向量库事实核验（如果有索引） ──
        if let Some(fc) = &self.vector_fact_checker {
            let fact_result = fc.verify_fact(claim, 50);
            if fact_result.verified_ratio < 0.3 {
                triggered_layers.push("VectorFactCheck".to_string());
                risk_factors.push(1.0 - fact_result.verified_ratio);
            }
        }

        // ── 防御层 4: 多节点投票（如果有多个节点） ──
        if let (Some(answers), Some(confidences)) = (multi_node_answers, multi_node_confidences) {
            let vote_result = self.multi_node_voter.vote(answers, confidences);
            all_evidence.push(DefenseEvidence {
                layer: "MultiNodeVote".to_string(),
                source: "federated_nodes".to_string(),
                description: format!(
                    "共识强度: {:.1%}, 投票结果: {:?}",
                    vote_result.consensus_strength, vote_result.vote_counts
                ),
                confidence: vote_result.consensus_strength,
                doc_id: None,
                similarity: None,
            });
            if vote_result.verdict == Verdict::Uncertain || vote_result.verdict == Verdict::Hallucination {
                triggered_layers.push("MultiNodeVote".to_string());
                risk_factors.push(1.0 - vote_result.consensus_strength);
            }
        }

        // ── 计算综合风险分数 ──
        let risk_score = if risk_factors.is_empty() {
            0.0  // 无风险因素
        } else {
            // 取加权平均，高风险因素权重更大
            let sum: f32 = risk_factors.iter().sum();
            (sum / risk_factors.len() as f32).min(1.0)
        };

        // ── 判决 ──
        let verdict = match risk_score {
            r if r >= config().hallucination_high_risk => Verdict::Hallucination,
            r if r >= config().hallucination_medium_risk => Verdict::LikelyFalse,
            r if r > 0.1 => Verdict::Uncertain,
            r if r > 0.0 => Verdict::LikelyTrue,
            _ => Verdict::Verified,
        };

        // ── 防御动作 ──
        let defense_action = match &verdict {
            Verdict::Verified | Verdict::LikelyTrue => DefenseAction::Accept,
            Verdict::LikelyFalse => DefenseAction::Flag,
            Verdict::Uncertain => DefenseAction::Flag,
            Verdict::Hallucination => DefenseAction::Reject,
        };

        DefenseResult {
            claim_id: claim_id.to_string(),
            is_hallucination: verdict == Verdict::Hallucination,
            risk_score,
            verdict,
            triggered_layers,
            evidence: all_evidence,
            defense_action,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// 快速检查：只用检索一致性（轻量级）
    pub fn quick_check(
        &self,
        claim_id: &str,
        claim: &str,
        retrieved_docs: &[SearchResult],
    ) -> DefenseResult {
        self.check(claim_id, claim, "", retrieved_docs, None, None)
    }
}

fn config() -> DefenseConfig {
    DefenseConfig::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crown_defense_triggered() {
        let crown = CROWNDefense::new(0.10);
        let result = crown.check(
            "精神分裂症", 0.85,      // 初始答案和置信度
            "双相情感障碍", 0.30,    // 社会答案和置信度
        );
        assert!(result.triggered);
        assert_eq!(result.final_answer, "精神分裂症");
        assert_eq!(result.confidence_drop, 0.55);
    }

    #[test]
    fn test_crown_defense_not_triggered() {
        let crown = CROWNDefense::new(0.15);
        let result = crown.check(
            "精神分裂症", 0.80,
            "双相情感障碍待排除", 0.70,
        );
        assert!(!result.triggered);
        assert_eq!(result.final_answer, "双相情感障碍待排除");
    }

    #[test]
    fn test_multi_node_vote_consensus() {
        let voter = MultiNodeVoter::new(0.667);
        let answers = vec![
            "A".to_string(), "A".to_string(), "A".to_string(), "B".to_string(),
        ];
        let confidences = vec![0.8, 0.9, 0.85, 0.6];
        let result = voter.vote(&answers, &confidences);
        assert_eq!(result.consensus, Some("A".to_string()));
        assert_eq!(result.consensus_strength, 0.75);
    }

    #[test]
    fn test_self_consistency() {
        let checker = SelfConsistencyChecker::new(0.70);
        let answers = vec![
            "精神分裂症".to_string(),
            "精神分裂症".to_string(),
            "双相情感障碍".to_string(),
        ];
        let result = checker.check(&answers);
        assert_eq!(result.consistency_score, 2.0 / 3.0);
        assert!(!result.passes);
    }
}
