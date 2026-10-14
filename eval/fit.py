"""Phases 2-4 data verification and fitting, from cached collection rows.

Reads results/collect/rows.jsonl (step rows), verifies Phase 2 (every row
labeled, signal distributions), fits variants A and B on the fit block
(plan 4.5), computes ECE/Brier on the cal block, chooses the variant with
the lowest Brier on cal, and computes the risk-controlled lambda table on
the cal trajectories (plan 4.6). Writes results/calibration.json and
results/calibration_report.md.

Usage:
    python -m eval.fit [--rows results/collect/rows.jsonl]
"""

import argparse
import json
import os
from collections import defaultdict

from eval.metrics import brier, ece, reliability
from gate.calibrator import (
    LogisticRegression,
    TemperatureScaling,
    calibrator_from_json,
    raw_verbalized,
)
from gate.threshold import ALPHAS, choose_lambda, expected_approvals


def load_rows(path):
    steps, trajs = [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "traj_done":
                trajs[(r["origin_split"], r["task_id"], r["trial"])] = r
            else:
                steps.append(r)
    return steps, trajs


def ascii_hist(values, bins=10, width=40):
    if not values:
        return "(empty)"
    lo, hi = 0.0, 1.0
    counts = [0] * bins
    for v in values:
        i = min(int((v - lo) / (hi - lo) * bins), bins - 1)
        counts[i] += 1
    mx = max(counts)
    out = []
    for i, c in enumerate(counts):
        bar = "#" * max(int(c / mx * width), 1 if c else 0)
        out.append(f"  {lo + i/bins:.1f}-{lo + (i+1)/bins:.1f} {c:5d} {bar}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=os.path.join("results", "collect", "rows.jsonl"))
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    steps, trajs = load_rows(args.rows)

    # ---- Phase 2 verification ----
    unlabeled = [r for r in steps if r.get("correct") is None]
    print(f"{len(steps)} step rows, {len(trajs)} trajectories")
    if unlabeled:
        print(f"PHASE 2 VERIFICATION FAILED: {len(unlabeled)} rows without label")
        raise SystemExit(2)
    print("Phase 2 check: every row has a label")

    fit_rows = [r for r in steps if r["split"] == "fit"]
    cal_rows = [r for r in steps if r["split"] == "cal"]
    print(f"fit rows: {len(fit_rows)} ({sum(1 for r in fit_rows if not r['correct'])} wrong)")
    print(f"cal rows: {len(cal_rows)} ({sum(1 for r in cal_rows if not r['correct'])} wrong)")
    if not fit_rows or not cal_rows:
        raise SystemExit("need rows in both fit and cal blocks")

    # ---- Phase 3: calibrators ----
    ts = TemperatureScaling().fit(fit_rows)
    lr = LogisticRegression().fit(fit_rows)

    y_cal = [1.0 if r["correct"] else 0.0 for r in cal_rows]
    p_raw = list(raw_verbalized(cal_rows))
    p_ts = list(ts.predict_proba(cal_rows))
    p_lr = list(lr.predict_proba(cal_rows))
    metrics = {
        "raw": {"ece": ece(y_cal, p_raw), "brier": brier(y_cal, p_raw)},
        "A_temperature_scaling": {"ece": ece(y_cal, p_ts), "brier": brier(y_cal, p_ts)},
        "B_logistic_regression": {"ece": ece(y_cal, p_lr), "brier": brier(y_cal, p_lr)},
    }
    for k, v in metrics.items():
        print(f"{k}: ECE={v['ece']:.4f} Brier={v['brier']:.4f}")

    chosen_name = min(metrics, key=lambda k: metrics[k]["brier"])
    chosen = {"A_temperature_scaling": ts, "B_logistic_regression": lr, "raw": None}[
        chosen_name
    ]
    print(f"chosen variant (lowest Brier on cal): {chosen_name}")

    # ---- Phase 4: thresholds on cal trajectories ----
    by_traj = defaultdict(list)
    for r in cal_rows:
        by_traj[(r["origin_split"], r["task_id"], r["trial"])].append(r)
    steps_by_traj = {t: rows for t, rows in by_traj.items()}
    p_hat_by_traj = {
        t: (
            list(raw_verbalized(rows))
            if chosen is None
            else list(chosen.predict_proba(rows))
        )
        for t, rows in by_traj.items()
    }
    lambda_table = {}
    approvals_table = {}
    for alpha in ALPHAS:
        lam = choose_lambda(steps_by_traj, p_hat_by_traj, alpha)
        lambda_table[f"{alpha:g}"] = lam
        approvals_table[f"{alpha:g}"] = expected_approvals(
            steps_by_traj, p_hat_by_traj, lam
        )
        print(f"alpha={alpha:g}: lambda={lam:.4f} expected_approvals={approvals_table[f'{alpha:g}']:.2f}")

    out_json = {
        "rows_path": args.rows,
        "n_steps": len(steps),
        "n_fit": len(fit_rows),
        "n_cal": len(cal_rows),
        "metrics": metrics,
        "chosen_variant": chosen_name,
        "calibrator": chosen.to_json() if chosen is not None else None,
        "lambda_table": lambda_table,
        "expected_approvals_cal": approvals_table,
    }
    with open(os.path.join(args.out, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    rel = {
        name: reliability(y_cal, p)
        for name, p in [("raw", p_raw), ("A", p_ts), ("B", p_lr)]
    }
    report = [
        "# Calibration report (Phases 2-4)",
        "",
        f"Rows: {len(steps)} steps ({len(fit_rows)} fit, {len(cal_rows)} cal), "
        f"{len(trajs)} trajectories.",
        "",
        "## Signal distributions (cal block)",
        "verbalized:",
        ascii_hist([r["verbalized"] for r in cal_rows]),
        "agreement:",
        ascii_hist([r["agreement"] for r in cal_rows]),
        "",
        "## Calibration metrics on cal",
        "| variant | ECE | Brier |",
        "|---|---|---|",
    ]
    for k, v in metrics.items():
        report.append(f"| {k} | {v['ece']:.4f} | {v['brier']:.4f} |")
    report += [
        "",
        f"Chosen for the live run: **{chosen_name}** (lowest Brier on cal).",
        "",
        "## Thresholds (cal trajectories, conformal risk control)",
        "| alpha | lambda | expected approvals/traj |",
        "|---|---|---|",
    ]
    for alpha in ALPHAS:
        a = f"{alpha:g}"
        report.append(
            f"| {alpha:g} | {lambda_table[a]:.4f} | {approvals_table[a]:.2f} |"
        )
    with open(os.path.join(args.out, "calibration_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"wrote {args.out}/calibration.json and calibration_report.md")


if __name__ == "__main__":
    main()
