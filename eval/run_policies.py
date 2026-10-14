"""Phase 6 runner: the three policies live on the test split.

For each policy (and each alpha for the calibrated one) runs every test
task num_trials times with the TauGateAgent and the perfect oracle, and
records per trajectory: executed_wrong_writes, approvals_requested,
reward, and cost. confirm_all skips signal computation (policy property
needs_signals=False); fixed and calibrated compute signals on gated steps.

Resumable like collect.py (traj markers with kind="traj_done").

Usage:
    python -m eval.run_policies --policy confirm_all --trials 4 --concurrency 4
    python -m eval.run_policies --policy calibrated --alphas 0.05 0.20
"""

import argparse
import json
import os
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from gate import config, llm
from gate.middleware import TauGateAgent
from gate.policy import make_policy

OUT_DIR = os.path.join("results", "policies")
# endpoint assumed for rows without the field (the Anthropic-compatible default)
DEFAULT_ENDPOINT = "https://api.z.ai/api/anthropic"
_write_lock = __import__("threading").Lock()


def run_one(policy_name, lam, task_id, trial, run_id, max_num_steps, calibrator=None):
    from tau_bench.envs.retail.env import MockRetailDomainEnv

    env = MockRetailDomainEnv(
        user_strategy="llm",
        user_model=config.USER_MODEL,
        user_provider=config.PROVIDER,
        task_split="test",
        task_index=task_id,
    )
    agent = TauGateAgent(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model=config.AGENT_MODEL,
        provider=config.PROVIDER,
        policy=make_policy(policy_name, lam),
        calibrator=calibrator if policy_name == "calibrated" else None,
    )
    agent.split_label = "test"
    agent.trial = trial
    result, records = agent.solve(env, task_index=task_id, max_num_steps=max_num_steps)
    return {
        "kind": "traj_done",
        "policy": policy_name,
        "lam": lam,
        "task_id": task_id,
        "trial": trial,
        "reward": result.reward,
        "executed_wrong_writes": agent.stats["executed_wrong_writes"],
        "approvals_requested": agent.stats["approvals_requested"],
        "rejections": agent.stats["rejections"],
        "gated_steps": len(records),
        "n_messages": len(result.messages),
        "run_id": run_id,
        "endpoint": config.ENDPOINT,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True,
                    choices=["confirm_all", "fixed_threshold", "calibrated"])
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.02, 0.05, 0.10, 0.20],
                    help="for the calibrated policy; lambda comes from the "
                    "threshold table produced by Phase 4")
    ap.add_argument("--calibration-file",
                    default=os.path.join("results", "calibration.json"))
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--max-num-steps", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from gate.threshold import ALPHAS  # noqa: F401  (documented sweep)

    runs = []  # (policy_name, lam or None)
    calibrator = None
    if args.policy == "calibrated":
        from gate.calibrator import calibrator_from_json

        with open(args.calibration_file, encoding="utf-8") as f:
            cal = json.load(f)
        if cal.get("calibrator") is None:
            print("ERROR: chosen variant is raw (no calibrator); the "
                  "calibrated policy needs a fitted variant", flush=True)
            sys.exit(2)
        calibrator = calibrator_from_json(cal["calibrator"])
        for alpha in args.alphas:
            lam = cal["lambda_table"].get(f"{alpha:g}")
            if lam is None:
                print(f"ERROR: no lambda for alpha={alpha} in {args.calibration_file}; "
                      "run Phase 4 (eval.fit) first", flush=True)
                sys.exit(2)
            if ("calibrated", lam) in runs:
                print(f"alpha={alpha} shares lambda={lam} with a smaller alpha; "
                      "running it once", flush=True)
                continue
            runs.append(("calibrated", lam))
    else:
        runs.append((args.policy, None))

    out = args.out or os.path.join(OUT_DIR, "rows.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    llm.install()
    unpriced = [m for m in (config.AGENT_MODEL, config.USER_MODEL) if m not in config.PRICES]
    if unpriced and os.environ.get("GATE_ALLOW_UNPRICED") != "1":
        print(f"ERROR: no price configured for {unpriced}", flush=True)
        sys.exit(2)

    run_id = uuid.uuid4().hex[:8]
    manifest = {
        "run_id": run_id,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": sys.argv[1:],
        "agent_model": config.AGENT_MODEL,
        "user_model": config.USER_MODEL,
        "provider": config.PROVIDER,
        "endpoint": config.ENDPOINT,
        "runs": [list(r) for r in runs],
        "trials": args.trials,
    }
    manifest_path = os.path.join(os.path.dirname(out), f"manifest_{run_id}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # resume: (policy, lam, task_id, trial), per endpoint: runs on a
    # different endpoint do not count as done (rows without the field come
    # from the default Anthropic-compatible endpoint)
    done = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if (r.get("endpoint") or DEFAULT_ENDPOINT) != (config.ENDPOINT or DEFAULT_ENDPOINT):
                        continue
                    done.add((r["policy"], r.get("lam"), r["task_id"], r["trial"]))
                except Exception:
                    pass

    jobs = []
    for pol, lam in runs:
        for tid in range(115):
            for tr in range(args.trials):
                if (pol, lam, tid, tr) not in done:
                    jobs.append((pol, lam, tid, tr))
    if args.max_tasks:
        jobs = jobs[: args.max_tasks]
    print(f"{len(jobs)} trajectories to run", flush=True)

    n_done = 0
    errors = []
    t0 = time.time()

    def attempt(pol, lam, tid, tr):
        try:
            return run_one(pol, lam, tid, tr, run_id, args.max_num_steps, calibrator)
        except llm.BudgetExceeded:
            raise
        except Exception:
            time.sleep(20)
            return run_one(pol, lam, tid, tr, run_id, args.max_num_steps, calibrator)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {
            ex.submit(attempt, pol, lam, tid, tr): (pol, lam, tid, tr)
            for pol, lam, tid, tr in jobs
        }
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                row = fut.result()
                with _write_lock:
                    with open(out, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_done += 1
                snap = llm.tracker.snapshot()
                print(
                    f"[{len(errors)}err] {key[0]}/lam={key[1]}/{key[2]}/tr{key[3]} "
                    f"reward={row['reward']:.0f} wrong={row['executed_wrong_writes']} "
                    f"asks={row['approvals_requested']} usd={snap['_total_usd']:.2f} "
                    f"({time.time()-t0:.0f}s)",
                    flush=True,
                )
            except llm.BudgetExceeded as e:
                print(f"BUDGET STOP: {e}", flush=True)
                for o in futs:
                    o.cancel()
                break
            except Exception as e:
                errors.append({"job": list(key), "error": repr(e),
                               "tb": traceback.format_exc()[-2000:]})
                print(f"ERROR {key}: {e!r}", flush=True)

    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["trajectories_run"] = n_done
    manifest["errors"] = errors
    manifest["usage"] = llm.tracker.snapshot()
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest["usage"], indent=2), flush=True)
    print(f"manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
