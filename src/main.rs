//! 联邦智能计算平台 - 入口
//!
//! 提供:
//! - CLI 交互界面
//! - REST HTTP API (Axum)
//! - gRPC 服务（tonic）

use federated_ai_platform::*;
use std::collections::HashMap;
use clap::Parser;

// ──────────────────────────────────────────────────────────────
// CLI 参数
// ──────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "federated-ai-platform")]
#[command(about = "联邦智能计算平台 — 自研HNSW向量库 · 联邦FL · RAG · CoT · Agent · 幻觉防御")]
struct Args {
    /// 启动 REST API 服务器
    #[arg(long, default_value_t = false)]
    serve,

    /// HTTP 监听地址
    #[arg(long, default_value = "0.0.0.0:8080")]
    listen: String,

    /// 向量库维度
    #[arg(long, default_value_t = 128)]
    dimension: usize,

    /// 交互式 CLI 模式
    #[arg(long, short, default_value_t = false)]
    interactive,

    /// 运行演示场景
    #[arg(long, short)]
    demo: Option<String>,
}

// ──────────────────────────────────────────────────────────────
// 演示场景
// ──────────────────────────────────────────────────────────────

fn run_demo_scenario(name: &str) {
    println!("\n{}", "═".repeat(70));
    println!("  🏛️  联邦智能计算平台 — 演示场景: {}", name);
    println!("{}", "═".repeat(70));

    match name {
        "hedge-fund" => demo_hedge_fund(),
        "medical" => demo_medical(),
        "agent" => demo_agent(),
        "hallucination" => demo_hallucination_defense(),
        "audit" => demo_audit(),
        "all" => {
            demo_hedge_fund();
            demo_agent();
            demo_hallucination_defense();
            demo_audit();
        },
        _ => {
            println!("未知场景: {}\n可用: hedge-fund, medical, agent, hallucination, audit, all", name);
        }
    }
}

fn demo_hedge_fund() {
    println!("\n📊 场景: 对冲基金 — 联邦策略分析\n");

    // ── 自研 HNSW 向量库 ──
    let mut index: vector_db::HnswVectorIndex<16, 32> = vector_db::HnswVectorIndex::new(16);

    // 插入基金风险画像向量
    let funds = vec![
        ("HF_ALPHA_001", vec![0.8, 0.9, 0.3, 0.7, 0.6, 0.5, 0.4, 0.9, 0.2, 0.1, 0.3, 0.8, 0.7, 0.5, 0.6, 0.4]),
        ("HF_BETA_002",  vec![0.6, 0.5, 0.8, 0.3, 0.2, 0.7, 0.9, 0.4, 0.5, 0.8, 0.9, 0.3, 0.2, 0.7, 0.8, 0.6]),
        ("HF_MACRO_003", vec![0.9, 0.7, 0.4, 0.8, 0.9, 0.3, 0.2, 0.6, 0.7, 0.4, 0.5, 0.9, 0.8, 0.3, 0.4, 0.7]),
        ("HF_CTA_004",   vec![0.3, 0.2, 0.9, 0.1, 0.4, 0.9, 0.7, 0.2, 0.9, 0.6, 0.3, 0.1, 0.4, 0.9, 0.6, 0.3]),
    ];

    for (id, vec) in funds {
        index.insert(id, vec).unwrap();
    }

    println!("  ✅ 向量库: {} 支基金, dim=16, 版本 v{}", index.size(), index.version());
    println!("  🔒 审计哈希: {}", index.snapshot_hash());

    // 搜索相似基金
    let query = vec![0.7, 0.8, 0.4, 0.7, 0.8, 0.4, 0.3, 0.7, 0.3, 0.3, 0.4, 0.7, 0.6, 0.5, 0.5, 0.5];
    let results = index.search(&query, 3, 50).unwrap();

    println!("\n  🔍 查询向量相似基金 (k=3):");
    for (i, r) in results.iter().enumerate() {
        println!("    {}. {} | 距离: {:.4f} | 相似度: {:.1%}",
            i + 1, r.id, r.distance, r.similarity());
    }

    // ── 审计链记录 ──
    let mut audit_chain = audit::AuditChain::new(10);
    audit::audit!(audit_chain, audit::EventType::FlNodeRegistered, "hf_node_alpha";
        domain = "hedge_fund", data_size = 5000);
    audit::audit!(audit_chain, audit::EventType::FlUpdateReceived, "hf_node_alpha";
        round = 1, weights_hash = "a3f8c...");
    audit::audit!(audit_chain, audit::EventType::FlAggregationCompleted, "server";
        round = 1, participants = 4, merkle_root = "e9b2d...");

    println!("\n  🔗 审计链验证: {}", if audit_chain.verify().is_ok() { "✅ 通过" } else { "❌ 失败" });
    println!("  📋 审计记录: {} 条", audit_chain.len());
}

fn demo_medical() {
    println!("\n🏥 场景: 医疗机构 — 联邦医学影像分析\n");

    let mut audit_chain = audit::AuditChain::new(10);
    audit::audit!(audit_chain, audit::EventType::FlNodeRegistered, "hospital_suzhou";
        institution = "苏州医院", data_size = 8000);
    audit::audit!(audit_chain, audit::EventType::FlNodeRegistered, "hospital_shanghai";
        institution = "上海胸科医院", data_size = 12000);
    audit::audit!(audit_chain, audit::EventType::FlAggregationCompleted, "med_fl_server";
        round = 3, participants = 2, model_auc = 0.923, privacy_mechanism = "local_dp");
    audit::audit!(audit_chain, audit::EventType::HallucinationDetected, "ai_assistant";
        claim = "该结节为恶性", confidence = 0.34, defense_action = "reject");

    println!("  ✅ 联邦学习训练完成: 3轮, 2家医院参与");
    println!("  🔐 隐私保护: 本地差分隐私 (ε=2.0)");
    println!("  📊 模型性能: AUROC = 0.923");
    println!("  🛡️ 幻觉防御: 1次检测触发（置信度不足被拒绝）");
    println!("  🔗 审计链长度: {} 条", audit_chain.len());
}

fn demo_agent() {
    println!("\n🤖 场景: 智能体 — 多工具协同分析\n");

    let mut agent = agent::create_demo_agent();

    let tasks = vec![
        "分析对冲基金 HF_ALPHA_001 的风险调整后收益",
        "比较三种不同市场环境下的策略表现",
        "如何优化私募基金的因子暴露？",
    ];

    for task in &tasks {
        println!("\n  📋 任务: {}", task);
        println!("  {}", "-".repeat(50));

        let plan = agent.plan_only(task);
        println!("  📐 规划: {} 步", plan.total_estimated_steps);
        for step in &plan.steps {
            println!("    {}. {:?} — {}", step.step_id, step.action, step.reasoning);
        }

        let result = agent.run(task);
        println!("  {}", "─".repeat(50));
        println!("  ✅ 执行: {} 步, 置信度: {:.1%}, 耗时: {}ms",
            result.steps_taken.len(), result.overall_confidence, result.total_duration_ms);

        if !result.reflections.is_empty() {
            println!("  💭 反思: {} 条", result.reflections.len());
            for r in &result.reflections {
                if let Some(imp) = &r.improvement {
                    println!("    • [Step {}] {} → {}", r.step, r.question, imp);
                }
            }
        }
    }
}

fn demo_hallucination_defense() {
    println!("\n🛡️ 场景: 幻觉防御 — 五层检测体系\n");

    let config = hallucination::DefenseConfig::default();

    // ── 层1: CROWN 防御 ──
    let crown = hallucination::CROWNDefense::new(config.crown_delta);
    let crown_result = crown.check(
        "该患者为精神分裂症谱系障碍",
        0.88,
        "建议排除双相情感障碍",
        0.25,
    );
    println!("  【CROWN防御】");
    println!("    初始置信度: 0.880 | 社会置信度: 0.250");
    println!("    下跌: {:.3f} | δ={:.3}", crown_result.confidence_drop, crown_result.crown_delta);
    println!("    触发: {} | 最终答案: {}", if crown_result.triggered { "🛡️ 是" } else { "否" }, crown_result.final_answer);
    println!("    原因: {}", crown_result.reason);

    // ── 层2: 多节点投票 ──
    let voter = hallucination::MultiNodeVoter::new(config.consensus_threshold);
    let answers = vec![
        "Long/Short Equity".to_string(),
        "Long/Short Equity".to_string(),
        "Global Macro".to_string(),
        "Stat Arb".to_string(),
    ];
    let confidences = vec![0.85, 0.79, 0.62, 0.55];
    let vote_result = voter.vote(&answers, &confidences);
    println!("\n  【多节点投票】");
    println!("    投票: {:?}", vote_result.vote_counts);
    println!("    共识: {:?} ({:.0%})", vote_result.consensus, vote_result.consensus_strength);
    println!("    分歧: {:?}", vote_result.dissenting);
    println!("    判定: {:?}", vote_result.verdict);

    // ── 层3: 自洽性检测 ──
    let sc_checker = hallucination::SelfConsistencyChecker::new(config.self_consistency_threshold);
    let sc_result = sc_checker.check(&vec![
        "应该配置60%股票".to_string(),
        "建议60%股票配置".to_string(),
        "70%股票更优".to_string(),
    ]);
    println!("\n  【LLM自洽性检测】");
    println!("    一致性: {:.0%} | 阈值: {:.0%}", sc_result.consistency_score * 100.0, config.self_consistency_threshold * 100.0);
    println!("    通过: {} | 最常见答案: {:?}", if sc_result.passes { "✅" } else { "❌" }, sc_result.most_common_answer);

    // ── 综合引擎 ──
    println!("\n  【综合防御引擎】");
    let engine = hallucination::HallucinationDefenseEngine::new(config, None);
    let mock_results: Vec<_> = vec![
        vector_db::SearchResult {
            id: "doc001".to_string(),
            distance: 0.12,
            metadata: HashMap::new(),
            version: 1,
        },
        vector_db::SearchResult {
            id: "doc002".to_string(),
            distance: 0.25,
            metadata: HashMap::new(),
            version: 1,
        },
    ];
    let def_result = engine.check(
        "claim_001",
        "联邦学习可以有效保护数据隐私",
        "关于联邦学习的论述",
        &mock_results,
        Some(&answers),
        Some(&confidences),
    );
    println!("    风险评分: {:.2} | 判定: {:?}", def_result.risk_score, def_result.verdict);
    println!("    触发层: {:?}", def_result.triggered_layers);
    println!("    防御动作: {:?}", def_result.defense_action);
}

fn demo_audit() {
    println!("\n🔗 场景: 区块链审计 — 全流程演示\n");

    let mut chain = audit::AuditChain::new(5);  // Merkle 批次大小 = 5

    // 模拟联邦学习全流程审计
    let events = vec![
        (audit::EventType::FlNodeRegistered, "node_a", "domain=hedge_fund,data_size=5000"),
        (audit::EventType::FlNodeRegistered, "node_b", "domain=hedge_fund,data_size=8000"),
        (audit::EventType::FlUpdateReceived, "node_a", "round=1,weights_hash=a3f8c..."),
        (audit::EventType::FlUpdateReceived, "node_b", "round=1,weights_hash=b7d2e..."),
        (audit::EventType::FlAggregationCompleted, "server", "round=1,merkle_root=e9b2d..."),
        (audit::EventType::RagQueryExecuted, "node_a", "query_id=q001,docs_returned=3"),
        (audit::EventType::HallucinationChecked, "defense_engine", "claim_id=c001,risk=0.28"),
        (audit::EventType::CotVoteRecorded, "cot_server", "task_id=t001,participants=3"),
    ];

    for (event_type, node_id, meta_str) in events {
        let mut meta = HashMap::new();
        for pair in meta_str.split(',') {
            let kv: Vec<_> = pair.split('=').collect();
            if kv.len() == 2 {
                meta.insert(kv[0].to_string(), kv[1].to_string());
            }
        }
        audit::audit!(chain, event_type, node_id; _meta = meta);
    }

    println!("  创世哈希: {}", chain.genesis());
    println!("  链长度: {} 条记录", chain.len());
    println!("  最新哈希: {}", chain.latest().unwrap().hash);

    // Merkle 验证
    if let Some(latest) = chain.latest() {
        if let Some(proof) = &latest.merkle_proof {
            println!("  Merkle证明: 高度={}, {} 个兄弟节点",
                proof.tree_height, proof.sibling_hashes.len());
        }
    }

    // 链完整性验证
    println!("\n  链完整性: {}", if chain.verify().is_ok() { "✅ 验证通过" } else { "❌ 链已损坏" });

    // 查询演示
    let fl_events = chain.query(Some(audit::EventType::FlAggregationCompleted), None, 10);
    println!("  FL聚合查询: {} 条", fl_events.len());
    for e in &fl_events {
        println!("    [{}] {} → {}", e.sequence, e.event_type, e.node_id);
    }
}

// ──────────────────────────────────────────────────────────────
// 交互式 CLI
// ──────────────────────────────────────────────────────────────

fn run_interactive() {
    use std::io::{self, Write};

    println!("\n{}", "═".repeat(60));
    println!("  🏛️  联邦智能计算平台 — 交互模式");
    println!("{}", "═".repeat(60));
    println!("\n输入任务描述，AI Agent 将自动规划并执行。");
    println!("命令: :plan <任务>  — 只规划不执行");
    println!("      :quit         — 退出");
    println!("      :audit        — 查看审计链");
    println!("\n─────────────────────────────────────────────────────\n");

    let mut agent = agent::create_demo_agent();
    let mut audit_chain = audit::AuditChain::new(10);

    loop {
        print!("🔮 > ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }
        let input = input.trim();
        if input.is_empty() { continue; }

        if input == ":quit" || input == "退出" {
            println!("再见！");
            break;
        }

        if input == ":audit" {
            println!("\n📋 审计链: {} 条记录", audit_chain.len());
            if let Some(l) = audit_chain.latest() {
                println!("最新: [{}] {} @ {}", l.sequence, l.event_type, l.node_id);
                println!("哈希: {}", l.hash);
            }
            println!();
            continue;
        }

        let (output, plan_mode) = if input.starts_with(":plan ") {
            let task = &input[6..];
            let plan = agent.plan_only(task);
            let plan_text = format!("📐 规划: {} 步\n{}",
                plan.total_estimated_steps,
                plan.steps.iter()
                    .map(|s| format!("  {}. {:?} — {}", s.step_id, s.action, s.reasoning))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
            (plan_text, true)
        } else {
            let result = agent.run(input);
            audit::audit!(audit_chain, audit::EventType::RagQueryExecuted, "cli_user";
                task_id = result.task_id.clone(), steps = result.steps_taken.len());
            let text = format!(
                "✅ 完成 ({} 步, {:.1%} 置信度, {}ms)\n\
                📋 执行步骤:\n{}\n\n💭 {}\n\n📊 最终答案:\n{}",
                result.steps_taken.len(),
                result.overall_confidence,
                result.total_duration_ms,
                result.steps_taken.iter()
                    .map(|s| format!("  • {:?} [{}ms] {}", s.tool, s.duration_ms,
                        if s.success { "✅" } else { "❌" }))
                    .collect::<Vec<_>>()
                    .join("\n"),
                if !result.reflections.is_empty() {
                    result.reflections.iter()
                        .filter_map(|r| r.improvement.as_ref().map(|i| format!("  💭 {}", i)))
                        .collect::<Vec<_>>()
                        .join("\n")
                } else {
                    "  (无反思)".to_string()
                },
                result.final_answer
            );
            (text, false)
        };

        println!("\n{}\n", output);
    }
}

// ──────────────────────────────────────────────────────────────
// 主入口
// ──────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    if let Some(demo) = args.demo {
        run_demo_scenario(&demo);
        return;
    }

    if args.interactive {
        run_interactive();
        return;
    }

    if args.serve {
        println!("🚀 启动 REST API 服务器: {}", args.listen);
        println!("   向量库维度: {}", args.dimension);
        // TODO: 实现完整的 Axum REST API
        // 实现 FastAPI/Rust HTTP server
        println!("\n⚠️  REST API 完整实现待完成。");
        println!("   目前可通过 --demo all 或 --interactive 测试。");
        return;
    }

    // 默认：打印概览
    println!("\n{}", "═".repeat(60));
    println!("  🏛️  联邦智能计算平台");
    println!("{}", "═".repeat(60));
    println!("\n用法:");
    println!("  --demo <场景>    运行演示场景");
    println!("    hedge-fund     对冲基金联邦策略分析");
    println!("    medical        医疗机构联邦学习");
    println!("    agent          智能体任务编排");
    println!("    hallucination  幻觉防御五层检测");
    println!("    audit          区块链审计");
    println!("    all            运行全部演示");
    println!("  --interactive, -i  交互式 CLI 模式");
    println!("  --serve           启动 REST API 服务器");
    println!();
    println!("运行演示:");
    println!("  cargo run -- --demo all");
    println!("  cargo run -- --demo hallucination");
    println!("  cargo run -- --interactive");
}
