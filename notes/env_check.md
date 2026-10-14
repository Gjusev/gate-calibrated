# Environment check (Phase 0)

Date: 2026-10-13. Machine: Windows 11, Git Bash.

## tau-bench source
- Repo: sierra-research/tau-bench, commit 59a200c6d575d595120f1cb70fea53cef0632f6b (2026-03-18). Local read-only clone in `vendor/tau-bench` (gitignored).
- Installed through `pyproject.toml` as a pinned git dependency.
- `setup.py` calls `open("README.md")` without an encoding. On Windows (cp1252) the install fails with UnicodeDecodeError. Workaround: set `PYTHONUTF8=1` before `pip install`. No vendor code was patched.

## Python and packages
- Python 3.11 venv in `.venv` (created with uv). `envs/retail/env.py` uses `match`, so Python >= 3.10 is required.
- litellm 1.105.0. Its price table has `claude-haiku-5-5` ($0.10 / $0.50 per Mtok in/out), `claude-sonnet-5-5` ($2 / $10), `claude-haiku-4-5`, `gpt-4o-mini`, `gpt-4.1-mini`. Not yet checked with a live call (no API keys in the environment).

## Retail splits (counted from the code)
| split | tasks | tasks with `outputs` | actions in gold lists |
|---|---|---|---|
| train | 500 | 0 | writes only, no read actions |
| dev | 20 | 0 | writes only |
| test | 115 | 37 | reads and writes |

Write-action counts in train: modify_pending_order_items 205, cancel_pending_order 177, exchange_delivered_order_items 158, return_delivered_order_items 146, modify_pending_order_address 48, modify_pending_order_payment 34, modify_user_address 0.
`modify_user_address` appears only in test (11 times). The calibrator never sees it in fit or cal.

## Tools (`tau_bench/envs/retail/tools/`)
Matches the plan's list, plus one tool the plan does not mention: **`think`** (writes nothing, returns ""). Classified as `special` (passes without gate). See notes/decisions_draft.md.

## Agent loop
`ToolCallingAgent.solve`: `completion(...)` -> `message_to_action(next_message)` -> `env.step(action)`, as the plan says. Only the first tool call of a message is executed (`tool_calls[:1]`).

## Plan vs code differences
1. Extra tool `think` (above).
2. `historical_trajectories/*-retail.json` are runs on the **test** split: gpt-4o has 460 entries (115 tasks x 4 trials), sonnet-35-new has 920 entries (115 tasks x 8 trials). Not a problem for Phase 1 (base error rate only), but these trajectories must never be used to fit or calibrate.
3. `dev` has 20 tasks. Conformal risk control needs N >= 1/alpha - 1 trajectories just for the bound to be reachable (alpha=0.02 needs N >= 49). `cal` therefore has to come from `train`, not `dev`.
4. `LLMUserSimulationEnv.__init__` calls `reset()`, which makes one paid model call when the env is built, before any task starts.
5. `LLMUserSimulationEnv.total_cost` is overwritten on each call, not accumulated. tau-bench's reported user cost is only the last call. Cost must be tracked separately (litellm callback).
6. `Env.calculate_reward` re-runs the gold actions through `self.step`, which appends them to `env.actions`. Harmless for reward, but anything reading `env.actions` after the end of an episode sees the gold actions too.

## Verification
- Free check: replayed the gold actions of every task through `MockRetailDomainEnv` (human user strategy, no model calls). Reward = 1.0 on 500/500 train, 20/20 dev, 115/115 test.
- Pending: one live retail task with the base agent. Blocked on API keys, models and budget (rule 1).
