# Decisions

Written on 2026-10-14 by the implementing agent (Claude) at Youssef's request, from `notes/decisions_draft.md`, which keeps the dated history. Status: not yet reviewed line by line by Youssef. Where a decision was his, it says so.

## Scope

- **Domain and benchmark.** tau-bench (Sierra, MIT) retail, pinned to commit 59a200c as a git dependency, so `pip install -e .` works from scratch. tau2-bench stays out of scope.
- **Abstain mode** (the agent asks the user instead of a reviewer) is out of scope. Youssef's call.
- **Repository** `Gjusev/gate-calibrated`, under `C:\Code Main\gate-calibrated`. Youssef's call.

## Data

- **Split by task, never by step.** `fit` is 250 train tasks drawn with seed 0; `cal` is the other 250 train tasks plus the 20 dev tasks; `test` is the tau-bench test split (115 tasks), used only for the live evaluation. `dev` alone (20 tasks) is too small: conformal risk control at alpha 0.02 needs at least 49 calibration trajectories before its bound can be met.
- **Historical trajectories** (gpt-4o and sonnet-3.5) are on the test split. They are used only to measure the base rate of wrong writes, never to fit or calibrate.
- **`think` is tier `special`** (no gate): it writes nothing. The plan's tool list did not include it.
- **Labeling is canonical comparison against gold actions**, by tool: `exchange_delivered_order_items` stores sorted item lists, so its lists compare as sets; `modify_pending_order_items` applies pairs in order, so with duplicate item ids the pair order is kept.

## Models and endpoint

- **No spending cap.** Youssef, 2026-10-13: "no escatimes en tokens ni nada". `MAX_USD` stays at 1000 as a guard against a runaway loop, not as a budget.
- **Agent and simulated user: glm-5.3-flash**, through z.ai. In 20-task pilots glm-5.3 erred on 4% of gated actions (too few positives to fit a calibrator) and glm-5.3-flash on about 12%, at a tenth of the price. The signals were saturated for both.
- **Signals use the agent's own model**, as the plan specifies.
- **Phase 2 ran on the z.ai coding-plan endpoint; the main Phase 6 run used the regular z.ai API** for all five policy configurations, after a temperature-0 equivalence check. The partial coding-plan Phase 6 run is kept and reported separately. Youssef suggested the regular API for its rate limits.
- **Pilot data collected before the labeling fixes was archived, not mixed in.**

## Method

- **Two calibrators** (temperature scaling, logistic regression); the live run uses the one with the lowest Brier score on `cal` (logistic regression).
- **One run per distinct lambda.** Alpha 0.10 and 0.20 gave the same lambda (0.6678), so they share one set of 460 trajectories.
- **The headline uses the regular-API run only**, so endpoint and task order are never confounded.
- **Negative results are reported as they came out**, per the plan's honesty rule.

## Engineering

- **One wrapper for every model call** (`gate/llm.py`): disk cache for temperature-0 calls, token and USD accounting from a dated price table (litellm has no GLM prices), and a backoff ladder for rate limits. Balance-exhaustion errors fail fast.
- **No key in any file.** Launch scripts map `ZAI_API_KEY` into the child process environment.
- **tau-bench's reported user cost is ignored** (it overwrites instead of summing); the wrapper's accounting is used instead.

## Identity (logo, film, README)

- **Concept: a boom barrier**, chosen by Youssef from three options. It lifts on its own (act) or holds with an amber lamp lit (ask).
- **Overpass**, the open version of the US road-sign face, for the wordmark; ink on paper with one amber accent used only for ask (palette and contrast in `DESIGN.md`).
- **The film uses only real rows** from the repo's data, including the case where the gate lets a wrong action through, and says plainly that the calibrated gate did not beat asking every time per approval.
- **No music in the film**: the bundled tracks' redistribution license was not verified; the sound effects are Kenney's (CC0).
