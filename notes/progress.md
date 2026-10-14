# Progress

## Phase 0. Environment (complete)
Done:
- Read PLAN.md. Confirmed with Youssef: location, repo name, abstain out of scope.
- Cloned tau-bench (commit 59a200c), created Python 3.11 venv, wrote pyproject.toml with tau-bench as a pinned git dependency. `pip install -e ".[dev]"` succeeds (needs PYTHONUTF8=1 on Windows).
- Checked splits, tool names, agent loop, litellm version and model table. Details and six plan-vs-code differences in notes/env_check.md.

Verified:
- Gold-action replay gives reward 1.0 on all 635 retail tasks (no model calls).
- Live run (2026-10-13): train task 0, base `ToolCallingAgent`, agent glm-5.3 + simulated user glm-5.3-flash through the z.ai Anthropic-compatible endpoint (env vars only, no keys in files). **Reward 1.0**, 20 messages, 74.8 s, 9 agent calls (43.6K in / 946 out tokens), 8 user calls (5.8K in / 853 out). Artifact: results/phase0_smoke.json.

Decisions recorded in notes/decisions_draft.md (7-10): no spending cap (Youssef: "no escatimes en tokens ni nada"; MAX_USD=1000 stays as a runaway guard only), provider z.ai, agent+signals glm-5.3, user glm-5.3-flash, split fit=250/cal=270/test=115.

## Phase 1. Tiers and labeling (complete)
Done:
- `gate/types.py` (StepRecord, GateDecision), `gate/tiers.py` (tier_of, gated), `gate/labeling.py` (canonical, is_correct, write_actions_from_messages). 11 tests, all pass, including the mandatory gold-list replay test over 50 train + 50 test tasks.
- Canonicalization detail found while testing: `modify_pending_order_items` and `exchange_delivered_order_items` pair item_ids[i] with new_item_ids[i]; the DB state does not depend on list order (price diff is a sum, exchange stores sorted lists), but it DOES depend on the pairing (variant-product matching). Canonical form: sorted list of pairs. Cited in code comments.

Verified (results/phase1_base_rates.json):
- Historical trajectories, gated actions labeled against gold:
  - gpt-4o: 460 trajectories, 666 gated actions, 199 wrong (**29.9%**), 148/460 trajectories have >= 1 wrong gated action. Mean reward 0.604.
  - sonnet-3.5: 920 trajectories (8 trials per task, not 4; env_check updated), 1352 gated actions, 284 wrong (**21.0%**), 227/920 trajectories affected. Mean reward 0.692.
- Consistency: mean reward is above 1 - wrong-trajectory-rate for both models, as expected, because some wrong writes are rejected by tool validation without changing the DB.
- Base rate for the problem: about 1 in 3 to 1 in 5 gated actions proposed by a mid-strength agent is wrong.

## Phases 3-6 code (written while Phase 2 collects; no fitting on real data yet)
- Phase 3: `gate/calibrator.py` (A: temperature scaling, grid+refine; B: logistic regression, IRLS with L2, standardized features), `eval/metrics.py` (ECE equal-mass 10 bins, Brier, reliability, pass^k unbiased estimator). Synthetic-data tests pass: variant A recovers a known miscalibration temperature and improves ECE; B beats raw Brier; saturated verbalized does not crash and B still ranks via agreement.
- Phase 4: `gate/threshold.py` (conformal risk control, trajectory unit, unreachable bound -> lambda=1.0). Monte Carlo test (200 reps x 3 alphas) passes: average new-sample loss <= alpha, tail within sampling noise. NOTE (documented in the test docstring): the plan's literal verification criterion ("empirical rate <= alpha in the great majority of repetitions") is stronger than what conformal risk control guarantees (a bound in expectation); when the true risk equals alpha exactly, a finite new sample exceeds alpha about half the time. The test checks the guarantee that actually holds.
- Phase 5: `gate/policy.py` (confirm_all skips signal computation entirely; fixed reads raw verbalized; calibrated reads p_hat), `gate/oracle.py` (perfect oracle), `TauGateAgent` in `gate/middleware.py` (ask flow: rejection injects "action rejected by reviewer" and the agent replans; stats: approvals_requested, executed_wrong_writes, rejections), `gate_node` LangGraph integration with `GateGraphState` TypedDict + `examples/minimal_gate_graph.py`. Stub tests: oracle rejects the wrong action and approves the correct one; fixed_threshold executes a wrong confident write; interrupt/resume works (a plain `dict` state schema silently drops keys between nodes - found in the example, fixed with a TypedDict).
- Phase 6: `eval/run_policies.py` (resumable, loads the fitted calibrator and lambda table from results/calibration.json). Mechanical smoke passed: 1 confirm_all trajectory on test task 0 -> reward 1.0, approvals 1, executed wrong writes 0.
- `eval/fit.py`: Phase 2 verification (no unlabeled rows), fits A and B on the fit block, ECE/Brier on cal, chooses the lowest-Brier variant, computes the lambda table on cal trajectories, writes results/calibration.json + calibration_report.md with signal histograms.
- 61 tests, all passing. pyproject: testpaths pinned (bare `pytest` from the root no longer recurses into vendor/), langgraph in dev extras.

## Phase 2. Collection with signals (complete)
Done:
- `gate/config.py`, `gate/llm.py` (disk cache for temperature-0 calls, per-model token/USD accounting, runaway guard MAX_USD=1000, retry with backoff against z.ai rate limits), `gate/signals.py` (verbalized via structured tool call `report_probability`, free-text fallback; agreement k=5 temp 0.7), `gate/middleware.py` (InstrumentedAgent with the gate hook between message_to_action and env.step; CollectAgent records and always acts), `eval/collect.py` (resumable runner, traj_done markers, budget stop with drain, per-run manifest).
- Adversarial review round (workflow, 5 dimensions, 23 findings): confirmed defects fixed with regression tests - hallucinated tool names no longer crash the loop (base-agent parity), verbalized now structured output, resume no longer re-runs zero-row trajectories, exchange/modify list canonicalization per actual DB semantics (duplicates positional for modify), uninstall restores the real litellm.completion, cache read retries on the Windows rename race, unpriced models fail fast. One root-cause bug found by the mandatory gold-replay test itself: `action_key` dropped the tool name, so exchange actions were never labeled correct against gold (52 tests, all pass).
- Pilots (archived): glm-5.3 agent -> 4% wrong gated actions (too few positives), verbalized/agreement saturated at 1.0. glm-5.3-flash -> 12% wrong, signals also saturated. Decision 11 in decisions_draft: agent = glm-5.3-flash.
- Full collection: 520 tasks (fit 250 train + cal 250 train + 20 dev) x 2 trials = 1040 trajectories, 0 errors, concurrency 8 after the first 29. Output results/collect/rows.jsonl. Cost $8.63 at list prices.

Note on phase order: Phase 3 and 4 code (calibrator, threshold, metrics) was written while collection runs, with synthetic-data tests only; no fitting on real data happens before the Phase 2 verification passes.

Findings so far (real, per plan honesty rules):
- z.ai rate-limits at concurrency 4 without retries (17/20 pilot trajectories lost once; fixed with backoff).
- tau-bench's reported user cost is wrong (overwritten per call) - our accounting is the tracker, not SolveResult.total_cost.
- verbalized confidence of GLM models on tau-bench is saturated at ~1.0 regardless of agent strength; agreement at k=5, temp 0.7 is 1.0 on most steps. The calibration results must be read with this in mind; it is the finding the plan anticipated in 4.4.

Verified (2026-10-14, `python -m eval.fit`, results/calibration_report.md):
- Every one of the 1535 step rows has a label. fit block 755 gated steps (43 wrong, 5.7%), cal block 780 (53 wrong, 6.8%).
- Signal histograms (cal): verbalized is in [0.9, 1.0] for 780 of 780 steps, including all 53 wrong ones. Agreement is in [0.9, 1.0] for 758 of 780. Both signals are saturated. Recorded as the finding the plan anticipated in 4.4.

## Phase 3. Calibrator (complete)
Verified on cal (fit on fit):
| variant | ECE | Brier |
|---|---|---|
| raw verbalized | 0.0659 | 0.0675 |
| A temperature scaling | 0.0531 | 0.0678 |
| B logistic regression | 0.0502 | 0.0647 |
- B is chosen (lowest Brier). Calibration improves ECE for both variants; A does not improve Brier (0.0678 vs raw 0.0675). Reported as is.
- What B learned (weights on standardized features): logit(verbalized) 0.03, agreement 0.12, irreversible -0.41, log1p(step_index) -0.80, bias 3.11. The model's own confidence carries almost no weight; position in the conversation (later steps are riskier) is the strongest feature. Write-step p_hat on cal ranges 0.79 to 0.99, median 0.97.

## Phase 4. Threshold (complete)
Verified on cal trajectories (conformal risk control, trajectory loss):
| alpha | lambda | expected approvals/traj (cal) |
|---|---|---|
| 0.02 | 0.9603 | 1.16 |
| 0.05 | 0.9136 | 1.01 |
| 0.10 | 0.6678 | 0.95 |
| 0.20 | 0.6678 | 0.95 |
- alpha 0.10 and 0.20 give the same lambda: at 0.6678 no write step is below lambda (min write p_hat is 0.79), so the calibrated policy asks only on irreversible actions, and the loss of that policy on cal already meets 0.10. Run once in Phase 6.
- What each lambda does on cal write steps (271, 31 wrong): lambda 0.668 asks on 0 and catches 0 wrong; 0.914 asks on 29 and catches 6; 0.960 asks on 111 (41%) and catches 22 of 31 (71%; random asking at 41% would catch about 13).
- Consequence for Phase 6: fixed_threshold (verbalized < 0.7) also never asks on a write step, because verbalized is never below 0.9. fixed_threshold and calibrated at lambda 0.668 are the same policy in practice; their difference in Phase 6 measures noise.

## Phase 5. Policies, oracle, middleware (complete)
See "Phases 3-6 code" above: stub integration tests (oracle rejects the wrong action, approves the correct one) and the LangGraph interrupt/resume test pass.

## Phase 6. Live evaluation (complete)
- First pass (2026-10-14) on the coding-plan endpoint, then the remaining runs on the regular z.ai API after a temperature-0 equivalence check (30 real conversation prefixes cut before a gated action: same-endpoint repeat agreement 14/30, cross-endpoint 18/30 and 15/30; share proposing a gated action 0.57/0.57 vs 0.60). Side finding: temperature 0 is not deterministic on this endpoint (14/30 identical next actions on a repeat), which is why the disk cache matters for reproducing reports.
- The switch confounded endpoint with task order (ai_review_log case 10), so Phase 6 is being completed in full on the regular API for all five configurations. The main report uses only that run. The coding-plan run is kept and reported separately.
- Plan verification already met on the coding-plan run: confirm_all, 460 trajectories, 0 executed wrong writes, 1.61 approvals per trajectory, 101 rejections.
- Regular-API run completed 2026-10-14 17:25 after a recharge: 2300/2300 trajectories (confirm_all, fixed_threshold, calibrated at lambda 0.9603/0.9136/0.6678), 0 errors. Cost: $57.16 measured for 2158 trajectories; 142 more from five processes stopped at the balance exhaustion have no usage record (about $3 to $4). The earlier $58.18 figure wrongly included a $1.02 coding-plan manifest. All conformal targets held live: wrong-write trajectories 1.7% at alpha 0.02 (target 2%), 3.7% at 0.05, 3.0% at 0.10/0.20. confirm_all: 0 of 460, 1.57 approvals/traj. fixed_threshold 4.6% at 0.95 approvals. fixed_threshold and calibrated at lambda 0.668 are the same policy in practice (over the 549 collected write steps they would ask on 1 and 0), so their gap (3.0% vs 4.6% wrong-write trajectories, pass^4 0.852 vs 0.826) measures noise, not an effect. Calibrated at lambda 0.960 is the only configuration that changes behavior on writes; per extra approval its rate of removing wrong writes is indistinguishable from confirm_all's (0.058 vs 0.062 wrong writes per approval against the pooled irreversible-only baseline; difference CI [-0.04, 0.03], indistinguishable). Task success gaps are mostly within noise; one exception, calibrated alpha 0.05 at 0.070 below the lambda 0.668 run on pass^4 (CI [-0.130, -0.017], uncorrected for about ten comparisons), is more likely noise than an effect but cannot be ruled out. This is the negative result the plan asks to report as is. Report: results/report.md, results/curve.png. The coding-plan confirm_all run is reported separately in the report's endpoint table (0 wrong writes there as well).
- Balance ran out mid-run at 15:27; the retry ladder treated it as transient and spun for ~20 minutes. Fixed: fail fast on balance errors (ai_review_log case 11, test_insufficient_balance_fails_fast).

## Phase 7. Report and write-up (complete)
- `python -m eval.report` (2026-10-14): results/report.md, curve.png, curve.json. All five configurations n=460, bootstrap CIs over tasks. README main result, curve and pass^k tables, and cost table filled from the report and manifests (Phase 2 $8.36 measured for 1009 of 1040 trajectories; Phase 6 regular API $57.16 measured for 2158 of 2300, see README Cost). The first README draft overclaimed a calibrated-vs-fixed advantage (ai_review_log case 12); the main result and results sections were rewritten from the data, and the Limitations section extended (saturation, irreversible share of approvals, endpoint). 62 tests pass. No git commits, per plan.
- Independent read-only review of the README (subagent, 2026-10-14) recomputed every claim from the raw rows and manifests: all table values, data claims and regression weights confirmed; four cost/wording errors and three overstatements found and fixed (ai_review_log case 13).
