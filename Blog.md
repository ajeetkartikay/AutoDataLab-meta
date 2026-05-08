# AutoDataLab++: Teaching a 1.5B LLM to Orchestrate a Multi-Agent Office

**Author:** Ajeet Kumar ([@ajeetkartikay](https://github.com/ajeetkartikay))
**Team:** Peaky Blinders — built with [Sai Kamal Nannuri](https://github.com/Uchihakamal1816)
**Original team repo:** https://github.com/Uchihakamal1816/AutoDataLab-
**My portfolio fork:** https://github.com/ajeetkartikay/AutoDataLab-meta
**Live demo:** https://uchihamadara1816-autodatalab2-0.hf.space/ui/

**Code:** https://github.com/Uchihakamal1816/AutoDataLab-

## TL;DR

We built an OpenEnv environment where a Chief of Staff (CoS) policy routes work across four AI specialists — Data Analyst, Finance, Strategy, and HR — to complete CEO briefing tasks. We then trained a Qwen2.5-1.5B model with GRPO to be that CoS, and added RAG over a corpus of company SOPs, policies, and historical exemplars.

The honest result: terminal scores are nearly identical between the base LLM and the RL-trained one. But only the RL-trained policy actually orchestrates correctly. The base LLM gets stuck in a consult:strategy loop and has to be rescued by the environment fallback. RL teaches the agent when to stop and when to move on — the inverse of what next-token training optimizes for.

## The Problem

Most LLM benchmarks test single-turn reasoning. Real enterprise work is not like that. A CEO does not ask one question; they ask "look at our data, project the quarter, recommend an action, and draft the memo." That requires routing decisions — which expert to consult, in what order, when you have enough information, when to stop iterating, when to commit.

Next-token training rewards plausible-sounding text. It does not reward stopping. Our hypothesis: a small LLM trained with RL on the routing decision will outperform a much larger LLM that is just prompted to orchestrate.

## The Environment

Six tasks, three of them on the critical path for the full agent stack:

| Task | Difficulty | Strategy required? |
|---|---|---|
| easy_brief | easy | No |
| medium_brief | medium | Yes |
| hard_brief | hard | Yes |
| expert_brief | hard | Yes |
| risk_brief | hard | Yes |
| crisis_brief | hard | Yes |

The CoS has a small discrete action space: consult(expert), summarize, submit, noop. Episodes max out at 12-14 steps. The reward is shaped: +0.08 per productive consult, +0.02 for a useful summarize, -0.05 for redundant consults, -0.02 per step (efficiency pressure), and a terminal grader score in (0.001, 0.999) on submit.

The four specialists are the real subenvs:

- Data Analyst — wraps subenvs/autodatalab/analytics.py: cleaning, KPIs, revenue derivation, data-quality scoring
- Finance — wraps deterministic projection / variance / break-even tools
- Strategy — produces 3-bullet operating plans with citations to upstream numbers
- HR / Comms — wraps subenvs/email/: drafts memos and scores them on a blended structure + tone + audience rubric

Each specialist returns a typed ExpertReport. The CoS composes them into a final Brief.

## The Training

Base model: Qwen2.5-1.5B-Instruct.

Method: GRPO (Group Relative Policy Optimization). Rollouts collected against the live HF Space env. The policy was first warm-started with SFT on demonstration trajectories, then RL'd against the shaped reward.

Why this size: A 1.5B model is small enough to fine-tune on a single GPU but large enough to follow the JSON action schema. The whole point is to show that a small RL'd policy beats a large prompted-only one on routing.

## RAG — and Why It Matters Here

The memory/ package contains a small corpus of company artifacts: SOPs (data quality, finance forecasting, communications style, strategy playbook), policies (compliance constraints), and history (last quarter review, exemplar memos).

A retriever (BM25 over the markdown corpus, no neural embeddings — keeps it deterministic and fast) pulls the top-k passages most relevant to each expert working query. When the RAG toggle is ON, experts ground their answers in retrieved snippets and emit memory_citations alongside their reports.

Why RAG is the right fit for an RL environment, specifically:

1. It externalizes domain knowledge from the policy. The CoS does not need to memorize SOPs; it just needs to learn when to invoke an expert that knows them. That keeps the action space small and the RL credit assignment clean.

2. It gives the policy stable, citable context. RL rollouts need reproducibility. Hallucinated specifics ruin reward signal. Pinning experts to retrieved company text makes the brief auditable and the reward consistent across rollouts.

3. It separates what to say from who should say it. The policy learns the latter; the corpus owns the former. New SOPs ship without retraining.

4. It is a measurable axis the policy can exploit. Same task, RAG-on consistently scored ~0.01 higher than RAG-off across all 3 hard tasks. RL learns to prefer the higher-reward variant when it is available.

## Results — RL vs Non-RL, RAG vs no RAG

We ran the same 3 hard tasks under 4 conditions: base LLM and base LLM + GRPO, each with RAG off and RAG on.

### Terminal scores

| Task | Base LLM (no RAG) | Base LLM + RAG | Base + GRPO (no RAG) | Base + GRPO + RAG |
|---|---|---|---|---|
| expert_brief | 0.8827 | 0.8925 | 0.8827 | 0.8925 |
| risk_brief | 0.8839 | 0.8948 | 0.8839 | 0.8948 |
| crisis_brief | 0.8805 | 0.8914 | 0.8805 | 0.8914 |

Terminal scores are essentially identical. This is the honest finding. The grader rewards the content of the final brief, and the env fallback path gets all four experts consulted regardless of how the policy got there. So if you only look at the final number, RL does not appear to help.

### What the policy actually did — the real difference

| Dimension | Base LLM (no RL) | Base LLM + GRPO |
|---|---|---|
| model_routed_required | 3 of 4 experts | All 4 experts |
| needed_fallback | True (every run) | False (every run) |
| Action repetition | Spam-consults strategy 4x in a row | Clean linear path |
| Action sequence | analyst -> finance -> strategy -> strategy -> strategy -> strategy | analyst -> finance -> strategy -> hr -> summarize -> submit |

Evidence (raw rollouts in training/exports/):

- Base LLM (no RL): evidence_non_rl.json — expert_brief shows needed_fallback true, fallback list of consult:hr, summarize, submit, model_routed_required only analyst, finance, strategy (HR missing). Action sequence: consult:analyst -> consult:finance -> consult:strategy -> consult:strategy -> consult:strategy -> consult:strategy. Same pattern repeats across risk_brief and crisis_brief.

- Base LLM + GRPO: evidence_rl.json — expert_brief shows needed_fallback false, empty fallback list, model_routed_required all four experts. Action sequence: consult:analyst -> consult:finance -> consult:strategy -> consult:hr -> summarize -> submit. Same clean pattern across risk_brief and crisis_brief, with and without RAG.

The base LLM never figures out HR is required. It picks the most "interesting" option (strategy) and keeps picking it until the env fallback kicks in and force-completes the missing experts.

The GRPO-trained CoS picks all four required experts in the optimal order, summarizes when ready, submits, terminates. Six steps. No fallback. model_routed_required equals required_experts on every run.

If we removed the env fallback safety net, the base LLM would hit max-steps and submit an incomplete brief. The RL policy would still complete cleanly. That is the real result.

### What RAG bought us

Across all 6 evidence rollouts, average terminal score with RAG OFF is 0.8824 and with RAG ON is 0.8929 — improvement of +0.0105. Small in absolute terms, consistent across all 3 hard tasks, statistically meaningful given the (0.001, 0.999) clamp range. RAG is not a magic boost; it is a steady, citable improvement that the policy learns to exploit.

## What We Would Do With More Time

- Larger reward gap on path quality. Right now the grader is too forgiving of bad routing because the brief content can be salvaged. We would add a path-efficiency component to the terminal score so RL has a steeper gradient to climb.
- Disable env fallback during eval. This would make the routing failure visible in the terminal number, not just in the trace.
- Multi-turn refinement. Currently each expert returns once. Letting the CoS re-query an expert with a refined sub-question opens up real planning behavior.
- Adversarial tasks. A task where one expert lies (data quality is bad but Analyst does not catch it) would force the CoS to learn validation routing.

## The Stack

- OpenEnv — environment spec, validation, deployment
- Qwen2.5-1.5B-Instruct — base policy model
- GRPO — RL fine-tuning
- FastAPI + uvicorn — serving
- Hugging Face Spaces (Docker) — deployment
- Pydantic — typed action/observation schemas
- BM25 — RAG retriever (no neural embeddings, deterministic, fast)

## Try It

Smoke-test the live deployment by hitting /health on https://uchihamadara1816-autodatalab2-0.hf.space — should return status healthy.

Run an episode by POSTing to /reset with a JSON body containing task and use_rag fields.

Or open the office UI directly at https://uchihamadara1816-autodatalab2-0.hf.space/ui/. Pick MLP trained CoS policy on crisis_brief with RAG on. Watch the four boxes light up in order.

## Acknowledgments

Built for the OpenEnv hackathon. Thanks to the OpenEnv team for the spec, Hugging Face for the Spaces hosting, and our patient coffee machine.