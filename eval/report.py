"""Phase 7: final report from cached runs.

Reads results/policies/rows.jsonl (per-trajectory policy outcomes, z.ai API,
OpenAI-compatible endpoint) and results/calibration.json, and writes into
results/:
  - curve.png / curve.json: executed wrong writes per trajectory vs approvals
    per trajectory, one point per policy configuration, with bootstrap 95%
    CIs over tasks (plan section 4.8, headline result)
  - report.md: headline table, pass^k, calibration and threshold tables

Usage:
    python -m eval.report
"""

import json
import os
import random
from collections import defaultdict

import matplotlib

from eval.metrics import pass_hat_k

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "results"


def load_policy_rows(path):
    """traj_done rows, first occurrence per (policy, lam, task, trial,
    endpoint). Duplicates can only appear if two processes ran the same
    job; keeping the first makes the report independent of that."""
    rows, seen, dup = [], set(), 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") != "traj_done":
                continue
            key = (r["policy"], r.get("lam"), r["task_id"], r["trial"], r.get("endpoint"))
            if key in seen:
                dup += 1
                continue
            seen.add(key)
            rows.append(r)
    if dup:
        print(f"dropped {dup} duplicate trajectory rows")
    return rows


def bootstrap_mean_ci(pairs, n_boot=10000, seed=0):
    """pairs: list of (task_id, value). Bootstrap over TASKS (plan 7)."""
    by_task = defaultdict(list)
    for tid, v in pairs:
        by_task[tid].append(v)
    tasks = list(by_task.values())
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [tasks[rng.randrange(len(tasks))] for _ in range(len(tasks))]
        vals = [v for grp in sample for v in grp]
        means.append(sum(vals) / len(vals))
    means.sort()
    flat = [v for grp in tasks for v in grp]
    return (
        sum(flat) / len(flat),
        means[int(0.025 * len(means))],
        means[int(0.975 * len(means))],
    )


def label_for(pol, lam, cal):
    if lam is None:
        return pol
    alphas = [a for a, l in cal["lambda_table"].items() if abs(l - lam) < 1e-9]
    return f"calibrated a={'/'.join(alphas)} (lam={lam:.3f})"


def build_curve(rows, cal):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["policy"], r.get("lam"))].append(r)
    order = {"confirm_all": 0, "fixed_threshold": 1, "calibrated": 2}
    curve = []
    for (pol, lam), rs in sorted(
        groups.items(), key=lambda kv: (order[kv[0][0]], -(kv[0][1] or 0))
    ):
        n = len(rs)
        wrong = bootstrap_mean_ci([(r["task_id"], float(r["executed_wrong_writes"])) for r in rs])
        appr = bootstrap_mean_ci([(r["task_id"], float(r["approvals_requested"])) for r in rs])
        anyw = bootstrap_mean_ci(
            [(r["task_id"], 1.0 if r["executed_wrong_writes"] > 0 else 0.0) for r in rs]
        )
        rew = defaultdict(list)
        for r in rs:
            rew[r["task_id"]].append(r["reward"])
        curve.append(
            {
                "policy": pol,
                "lam": lam,
                "label": label_for(pol, lam, cal),
                "n": n,
                "wrong_per_traj": round(wrong[0], 4),
                "wrong_ci": [round(wrong[1], 4), round(wrong[2], 4)],
                "approvals_per_traj": round(appr[0], 4),
                "approvals_ci": [round(appr[1], 4), round(appr[2], 4)],
                "trajs_with_wrong": round(anyw[0], 4),
                "trajs_with_wrong_ci": [round(anyw[1], 4), round(anyw[2], 4)],
                "pass_hat": {f"k{k}": round(pass_hat_k(rew, k), 4) for k in (1, 2, 3, 4)},
                "mean_reward": round(sum(r["reward"] for r in rs) / n, 4),
                "rejections_per_traj": round(sum(r["rejections"] for r in rs) / n, 4),
            }
        )
    return curve


def plot(curve, path, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"confirm_all": "tab:red", "fixed_threshold": "tab:orange", "calibrated": "tab:blue"}
    for i, c in enumerate(curve):
        xerr = [[c["approvals_per_traj"] - c["approvals_ci"][0]],
                [c["approvals_ci"][1] - c["approvals_per_traj"]]]
        yerr = [[c["wrong_per_traj"] - c["wrong_ci"][0]],
                [c["wrong_ci"][1] - c["wrong_per_traj"]]]
        ax.errorbar(c["approvals_per_traj"], c["wrong_per_traj"], xerr=xerr, yerr=yerr,
                    fmt="o", capsize=3, color=colors.get(c["policy"], "gray"))
        ax.annotate(c["label"], (c["approvals_per_traj"], c["wrong_per_traj"]),
                    textcoords="offset points", xytext=(6, 4 if i % 2 else -12), fontsize=8)
    ax.set_xlabel("approvals requested per trajectory")
    ax.set_ylabel("executed wrong writes per trajectory")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render(curve, cal, endpoint_desc, suffix, extra=()):
    lines = [
        "# gate-calibrated results",
        "",
        f"Endpoint: {endpoint_desc}. Agent and simulated user: glm-5.3-flash. "
        "Oracle: perfect (gold actions). Test split, 115 tasks x 4 trials per "
        "configuration where complete. Full configs in the run manifests.",
        "",
        "## Headline: wrong writes executed vs approvals requested",
        "",
        "Bootstrap 95% CIs resample tasks. The calibrated rows target "
        "P(trajectory has >= 1 executed wrong write) <= alpha.",
        "",
        "| policy | n | wrong writes/traj (95% CI) | trajs with >=1 wrong write (95% CI) | approvals/traj (95% CI) | rejections/traj |",
        "|---|---|---|---|---|---|",
    ]
    for c in curve:
        lines.append(
            f"| {c['label']} | {c['n']} | {c['wrong_per_traj']:.3f} "
            f"[{c['wrong_ci'][0]:.3f}, {c['wrong_ci'][1]:.3f}] | "
            f"{c['trajs_with_wrong']:.1%} [{c['trajs_with_wrong_ci'][0]:.1%}, {c['trajs_with_wrong_ci'][1]:.1%}] | "
            f"{c['approvals_per_traj']:.2f} [{c['approvals_ci'][0]:.2f}, {c['approvals_ci'][1]:.2f}] | "
            f"{c['rejections_per_traj']:.2f} |"
        )
    lines += ["", f"![curve](curve{suffix}.png)", ""]
    lines += [
        "## pass^k (unbiased estimator)",
        "",
        "| policy | pass^1 | pass^2 | pass^3 | pass^4 | mean reward |",
        "|---|---|---|---|---|---|",
    ]
    for c in curve:
        p = c["pass_hat"]
        lines.append(
            f"| {c['label']} | {p['k1']:.3f} | {p['k2']:.3f} | {p['k3']:.3f} | "
            f"{p['k4']:.3f} | {c['mean_reward']:.3f} |"
        )
    lines += ["", "## Calibration (cal block, fit on the fit block)", "",
              "| variant | ECE | Brier |", "|---|---|---|"]
    for k, v in cal["metrics"].items():
        lines.append(f"| {k} | {v['ece']:.4f} | {v['brier']:.4f} |")
    lines += [
        "",
        f"Chosen for the live run: {cal['chosen_variant']} (lowest Brier on cal).",
        "",
        "| alpha | lambda | expected approvals/traj (cal) |",
        "|---|---|---|",
    ]
    for a, lam in cal["lambda_table"].items():
        lines.append(f"| {a} | {lam:.4f} | {cal['expected_approvals_cal'][a]:.2f} |")
    lines += list(extra)
    with open(os.path.join(OUT, f"report{suffix}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return lines


def main():
    rows = load_policy_rows(os.path.join(OUT, "policies", "rows.jsonl"))
    cal = json.load(open(os.path.join(OUT, "calibration.json"), encoding="utf-8"))
    if not rows:
        raise SystemExit("no policy rows; run Phase 6 first")
    curve = build_curve(rows, cal)
    plot(curve, os.path.join(OUT, "curve.png"),
         "gate policies on tau-bench retail (test split)")
    with open(os.path.join(OUT, "curve.json"), "w", encoding="utf-8") as f:
        json.dump(curve, f, indent=2)
    lines = render(curve, cal, "z.ai API (OpenAI-compatible endpoint)", "")
    print(chr(10).join(lines))
    print()
    print("wrote results/report.md, curve.png, curve.json")


if __name__ == "__main__":
    main()
