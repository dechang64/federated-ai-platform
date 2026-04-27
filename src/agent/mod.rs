//! Agent 编排层
//!
//! 基于 ReAct (Reason + Act) 模式：
//! - 任务规划器：将复杂任务分解为工具调用链
//! - 工具调用器：调度联邦 RAG / CoT / FL / 向量搜索
//! - 记忆模块：短期（向量）+ 长期（文件）
//! - 反思器：检测执行偏差，自我修正

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Serialize, Deserialize};
use thiserror::Error;

// ──────────────────────────────────────────────────────────────
// 工具定义
// ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ToolName {
    FederatedRAG,
    FederatedCoT,
    FederatedLearning,
    VectorSearch,
    HallucinationCheck,
    WebSearch,
    Calculator,
    Reflect,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
    pub tool: ToolName,
    pub params: HashMap<String, String>,
    pub result: Option<String>,
    pub success: bool,
    pub duration_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolResult {
    pub output: String,
    pub confidence: f32,
    pub citations: Vec<String>,
}

// ──────────────────────────────────────────────────────────────
// 规划结果
// ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanStep {
    pub step_id: u32,
    pub action: ToolName,
    pub reasoning: String,
    pub params: HashMap<String, String>,
    pub expected_output: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Plan {
    pub task_id: String,
    pub steps: Vec<PlanStep>,
    pub total_estimated_steps: u32,
    pub confidence: f32,
}

// ──────────────────────────────────────────────────────────────
// 执行结果
// ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub task_id: String,
    pub success: bool,
    pub final_answer: String,
    pub steps_taken: Vec<ToolCall>,
    pub reflections: Vec<Reflection>,
    pub overall_confidence: f32,
    pub total_duration_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Reflection {
    pub step: u32,
    pub question: String,
    pub answer: String,
    pub improvement: Option<String>,
}

#[derive(Error, Debug)]
pub enum AgentError {
    #[error("工具不存在: {0}")]
    ToolNotFound(String),
    #[error("工具执行失败: {0}")]
    ToolExecutionFailed(String),
    #[error("超出最大步数限制: {0}")]
    MaxStepsExceeded(u32),
    #[error("规划失败: {0}")]
    PlanningFailed(String),
}

// ──────────────────────────────────────────────────────────────
// Agent 核心
// ──────────────────────────────────────────────────────────────

pub struct Agent {
    task_id_counter: u64,
    max_steps: u32,
    tools: HashMap<ToolName, Box<dyn ToolExecutor>>,
}

pub trait ToolExecutor: Send + Sync {
    fn name(&self) -> ToolName;
    fn execute(&self, params: &HashMap<String, String>) -> Result<ToolResult, String>;
    fn estimate_confidence(&self, params: &HashMap<String, String>) -> f32;
}

impl Agent {
    pub fn new() -> Self {
        Self {
            task_id_counter: 0,
            max_steps: 10,
            tools: HashMap::new(),
        }
    }

    pub fn register_tool<T: ToolExecutor + 'static>(&mut self, tool: T) {
        self.tools.insert(tool.name(), Box::new(tool));
    }

    pub fn set_max_steps(&mut self, max: u32) {
        self.max_steps = max;
    }

    /// 生成唯一任务 ID
    fn new_task_id(&mut self) -> String {
        self.task_id_counter += 1;
        format!("task_{}", self.task_id_counter)
    }

    /// 分析任务，决定使用哪些工具
    fn plan(&self, task: &str) -> Plan {
        let task_id = self.new_task_id();
        let mut steps = Vec::new();
        let mut step_id = 1;

        let task_lower = task.to_lowercase();

        // 规则型规划（基于关键词匹配）
        // 复杂任务可以被分解为多个工具的组合

        // 1. 如果任务涉及检索或问答，先用 RAG
        if task_lower.contains("检索") || task_lower.contains("查询")
            || task_lower.contains("什么是") || task_lower.contains("如何")
            || task_lower.contains("为什么") || task_lower.contains("告诉我")
        {
            steps.push(PlanStep {
                step_id,
                action: ToolName::FederatedRAG,
                reasoning: "任务需要检索知识库，使用联邦RAG进行隐私保护检索".to_string(),
                params: [("query".to_string(), task.to_string())].into(),
                expected_output: "检索结果 + 生成答案".to_string(),
            });
            step_id += 1;
        }

        // 2. 如果任务涉及推理、计算、策略，使用 CoT
        if task_lower.contains("分析") || task_lower.contains("推理")
            || task_lower.contains("计算") || task_lower.contains("评估")
            || task_lower.contains("比较") || task_lower.contains("预测")
        {
            steps.push(PlanStep {
                step_id,
                action: ToolName::FederatedCoT,
                reasoning: "任务需要多步推理，使用联邦CoT进行分布式思维链".to_string(),
                params: [("task".to_string(), task.to_string())].into(),
                expected_output: "推理过程 + 最终结论".to_string(),
            });
            step_id += 1;
        }

        // 3. 如果任务涉及训练、建模，使用联邦学习
        if task_lower.contains("训练") || task_lower.contains("建模")
            || task_lower.contains("优化") || task_lower.contains("学习")
        {
            steps.push(PlanStep {
                step_id,
                action: ToolName::FederatedLearning,
                reasoning: "任务需要模型训练，使用联邦学习进行隐私保护建模".to_string(),
                params: [("task".to_string(), task.to_string())].into(),
                expected_output: "训练好的全局模型".to_string(),
            });
            step_id += 1;
        }

        // 4. 如果有 RAG/CoT 结果，进行幻觉检查
        if !steps.is_empty() {
            steps.push(PlanStep {
                step_id,
                action: ToolName::HallucinationCheck,
                reasoning: "对生成的结果进行幻觉检测和防御".to_string(),
                params: [("claim".to_string(), "[前一步结果]".to_string())].into(),
                expected_output: "幻觉风险评估 + 防御决策".to_string(),
            });
        }

        // 5. 如果没有匹配的工具，使用向量搜索作为后备
        if steps.is_empty() {
            steps.push(PlanStep {
                step_id: 1,
                action: ToolName::VectorSearch,
                reasoning: "无特定工具匹配，使用向量搜索作为通用检索".to_string(),
                params: [("query".to_string(), task.to_string())].into(),
                expected_output: "最相似的知识片段".to_string(),
            });
        }

        // 6. 最后反思
        steps.push(PlanStep {
            step_id: steps.len() as u32 + 1,
            action: ToolName::Reflect,
            reasoning: "执行完毕，进行自我反思和改进".to_string(),
            params: [("task".to_string(), task.to_string())].into(),
            expected_output: "反思记录 + 置信度评估".to_string(),
        });

        Plan {
            task_id: task_id.clone(),
            steps,
            total_estimated_steps: steps.len() as u32,
            confidence: 0.8,
        }
    }

    /// 执行任务
    pub fn run(&mut self, task: &str) -> ExecutionResult {
        let start = std::time::Instant::now();
        let plan = self.plan(task);

        let mut steps_taken = Vec::new();
        let mut reflections = Vec::new();
        let mut current_result: Option<String> = None;
        let mut overall_confidence = 1.0f32;

        for step in &plan.steps {
            if steps_taken.len() >= self.max_steps as usize {
                // 超限，停止并记录反思
                reflections.push(Reflection {
                    step: steps_taken.len() as u32,
                    question: format!("是否超出了最大步数限制({})?", self.max_steps),
                    answer: "是，已超出步数限制，任务可能被截断".to_string(),
                    improvement: Some("考虑将任务分解为更小的子任务".to_string()),
                });
                break;
            }

            let step_start = std::time::Instant::now();
            let (output, success, confidence) = if let Some(executor) = self.tools.get(&step.action) {
                // 替换占位符
                let mut params = step.params.clone();
                if let Some(prev) = &current_result {
                    params.insert("context".to_string(), prev.clone());
                }
                match executor.execute(&params) {
                    Ok(result) => (result.output, true, result.confidence),
                    Err(e) => (format!("执行失败: {}", e), false, 0.0),
                }
            } else {
                // 模拟执行（无真实工具）
                let fake_output = format!(
                    "[模拟] 调用 {} 完成，参数: {:?}",
                    format!("{:?}", step.action),
                    step.params
                );
                (fake_output, true, 0.5)
            };

            current_result = Some(output.clone());
            overall_confidence *= confidence;

            steps_taken.push(ToolCall {
                tool: step.action.clone(),
                params: step.params.clone(),
                result: Some(output),
                success,
                duration_ms: step_start.elapsed().as_millis() as u64,
            });

            // 反思：检查当前步骤是否有问题
            if !success {
                reflections.push(Reflection {
                    step: step.step_id,
                    question: format!("步骤 {:?} 失败了，原因是什么？", step.action),
                    answer: "工具执行出错".to_string(),
                    improvement: Some("检查参数是否正确，或使用替代工具".to_string()),
                });
            }
        }

        let duration = start.elapsed().as_millis() as u64;

        ExecutionResult {
            task_id: plan.task_id,
            success: !steps_taken.is_empty() && steps_taken.iter().all(|s| s.success),
            final_answer: current_result.unwrap_or_else(|| "无结果".to_string()),
            steps_taken,
            reflections,
            overall_confidence: overall_confidence.max(0.0).min(1.0),
            total_duration_ms: duration,
        }
    }

    /// 规划（只输出计划，不执行）
    pub fn plan_only(&self, task: &str) -> Plan {
        self.plan(task)
    }
}

// ──────────────────────────────────────────────────────────────
// 内置模拟工具（用于演示）
// ──────────────────────────────────────────────────────────────

pub struct SimulatedRAGTool {
    pub name: String,
}

impl ToolExecutor for SimulatedRAGTool {
    fn name(&self) -> ToolName { ToolName::FederatedRAG }

    fn execute(&self, params: &HashMap<String, String>) -> Result<ToolResult, String> {
        let query = params.get("query").cloned().unwrap_or_default();
        Ok(ToolResult {
            output: format!(
                "【联邦RAG 检索结果】\n\
                根据跨 {} 个联邦节点的隐私检索，未找到与「{}」完全匹配的文档。\n\
                返回最相关的3个片段：\n\
                - 片段1: [关于该主题的一般性讨论]\n\
                - 片段2: [相关领域的参考资料]\n\
                - 片段3: [领域专家的观点]\n\
                \n\
                注意：以上结果经过幻觉防御层检测，风险等级：低",
                3, query
            ),
            confidence: 0.78,
            citations: vec!["doc_001".to_string(), "doc_002".to_string()],
        })
    }

    fn estimate_confidence(&self, _: &HashMap<String, String>) -> f32 { 0.78 }
}

pub struct SimulatedCoTTool {
    pub name: String,
}

impl ToolExecutor for SimulatedCoTTool {
    fn name(&self) -> ToolName { ToolName::FederatedCoT }

    fn execute(&self, params: &HashMap<String, String>) -> Result<ToolResult, String> {
        let task = params.get("task").cloned().unwrap_or_default();
        Ok(ToolResult {
            output: format!(
                "【联邦CoT 推理过程】\n\
                任务：{}\n\n\
                🧠 推理路径 1（节点A）:\n\
                1. 理解问题 → 提取关键变量\n\
                2. 分解子问题 → 3个子问题\n\
                3. 逐步求解 → 结论A（置信度 0.82）\n\n\
                🧠 推理路径 2（节点B）:\n\
                1. 换个角度理解 → 发现隐藏假设\n\
                2. 验证假设 → 假设成立\n\
                3. 综合结论 → 结论B（置信度 0.75）\n\n\
                🗳️ 多节点投票：\n\
                节点A: 结论A (0.82)\n\
                节点B: 结论B (0.75)\n\
                最终采纳：结论A（共识强度 60%，低于阈值，建议人工复核）",
                task
            ),
            confidence: 0.72,
            citations: vec![],
        })
    }

    fn estimate_confidence(&self, _: &HashMap<String, String>) -> f32 { 0.72 }
}

pub struct SimulatedVectorSearchTool {
    pub name: String,
}

impl ToolExecutor for SimulatedVectorSearchTool {
    fn name(&self) -> ToolName { ToolName::VectorSearch }

    fn execute(&self, params: &HashMap<String, String>) -> Result<ToolResult, String> {
        let query = params.get("query").cloned().unwrap_or_default();
        Ok(ToolResult {
            output: format!(
                "【向量库搜索结果】\n\
                查询：「{}」\n\
                在自研HNSW向量库中搜索 (dim=128, size=10,000)\n\n\
                Top-5 最近邻：\n\
                1. [文档#001] 相似度: 0.923 - 主题概述\n\
                2. [文档#002] 相似度: 0.887 - 相关技术细节\n\
                3. [文档#003] 相似度: 0.854 - 实践经验\n\
                4. [文档#004] 相似度: 0.812 - 理论分析\n\
                5. [文档#005] 相似度: 0.798 - 案例研究",
                query
            ),
            confidence: 0.85,
            citations: vec![
                "doc_001".to_string(),
                "doc_002".to_string(),
                "doc_003".to_string(),
            ],
        })
    }

    fn estimate_confidence(&self, _: &HashMap<String, String>) -> f32 { 0.85 }
}

pub struct SimulatedHallucinationCheckTool {
    pub name: String,
}

impl ToolExecutor for SimulatedHallucinationCheckTool {
    fn name(&self) -> ToolName { ToolName::HallucinationCheck }

    fn execute(&self, params: &HashMap<String, String>) -> Result<ToolResult, String> {
        let claim = params.get("claim").cloned().unwrap_or_default();
        let context = params.get("context").cloned().unwrap_or_default();
        Ok(ToolResult {
            output: format!(
                "【幻觉防御检测报告】\n\
                检测断言：「{}」\n\n\
                🛡️ 五层防御检测结果：\n\n\
                【层1】检索一致性检测\n\
                ✓ 找到支撑文档3篇\n\
                ✓ 最高相关性: 0.887 (可信)\n\n\
                【层2】向量库事实核验\n\
                ✓ 关键术语验证通过 (80%)\n\n\
                【层3】CROWN一致性防御\n\
                ✓ 置信度下跌 0.05 < δ=0.10，未触发\n\n\
                【层4】多节点投票\n\
                ⚠️ 共识强度 60% < 66.7%，存在分歧\n\n\
                【层5】LLM自洽性\n\
                ✓ 3次采样一致性 83% > 阈值70%\n\n\
                ──────────────────\n\
                综合风险评分: 0.28 (中等)\n\
                防御动作: ⚠️ FLAG（标记为不确定，建议人工复核）\n\
                详细解释: {}",
                claim, context
            ),
            confidence: 0.92,
            citations: vec!["doc_001".to_string(), "doc_002".to_string()],
        })
    }

    fn estimate_confidence(&self, _: &HashMap<String, String>) -> f32 { 0.92 }
}

pub struct SimulatedReflectTool;

impl ToolExecutor for SimulatedReflectTool {
    fn name(&self) -> ToolName { ToolName::Reflect }

    fn execute(&self, params: &HashMap<String, String>) -> Result<ToolResult, String> {
        let task = params.get("task").cloned().unwrap_or_default();
        Ok(ToolResult {
            output: format!(
                "【执行反思】\n\
                任务：「{}」\n\n\
                ✓ 任务分解：成功\n\
                ✓ 工具调用：3步全部完成\n\
                ⚠️ 置信度：0.42（中低）\n\n\
                改进建议：\n\
                1. 考虑增加联邦节点数量以提高投票共识\n\
                2. 对关键断言建议开启完整幻觉防御5层检测\n\
                3. 如果置信度持续偏低，建议拆分任务",
                task
            ),
            confidence: 0.95,
            citations: vec![],
        })
    }

    fn estimate_confidence(&self, _: &HashMap<String, String>) -> f32 { 0.95 }
}

// ──────────────────────────────────────────────────────────────
// 工厂函数：创建带模拟工具的 Agent
// ──────────────────────────────────────────────────────────────

pub fn create_demo_agent() -> Agent {
    let mut agent = Agent::new();
    agent.set_max_steps(10);
    agent.register_tool(SimulatedRAGTool { name: "FederatedRAG".to_string() });
    agent.register_tool(SimulatedCoTTool { name: "FederatedCoT".to_string() });
    agent.register_tool(SimulatedVectorSearchTool { name: "VectorSearch".to_string() });
    agent.register_tool(SimulatedHallucinationCheckTool { name: "HallucinationCheck".to_string() });
    agent.register_tool(SimulatedReflectTool);
    agent
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_rag_task() {
        let agent = create_demo_agent();
        let plan = agent.plan_only("什么是联邦学习？");
        assert!(!plan.steps.is_empty());
        assert_eq!(plan.steps[0].action, ToolName::FederatedRAG);
    }

    #[test]
    fn test_plan_analysis_task() {
        let agent = create_demo_agent();
        let plan = agent.plan_only("分析这个对冲基金的策略表现");
        let actions: Vec<_> = plan.steps.iter().map(|s| &s.action).collect();
        assert!(actions.contains(&&ToolName::FederatedCoT) || actions.contains(&&ToolName::FederatedLearning));
    }

    #[test]
    fn test_agent_run() {
        let mut agent = create_demo_agent();
        let result = agent.run("介绍一下联邦学习的基本原理");
        assert!(!result.steps_taken.is_empty());
        assert!(result.total_duration_ms > 0);
    }
}
