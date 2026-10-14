"""Phase 2 runner: base agent with signal collection over fit and cal.

For every gated action the agent proposes, both signals (verbalized,
agreement) and the label are recorded, then the action executes (base
behavior, no gate). Output: one JSONL row per gated action, plus a manifest
with the full config and cost accounting. Resumable: (origin_split,
task_id, trial) triples already in the output are skipped.

Usage:
    python -m eval.collect --split fit --trials 2 --concurrency 4
    python -m eval.collect --max-tasks 2 --trials 1   # smoke
"""

import argparse
import dataclasses
import json
import os
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from gate import config, llm
from gate.middleware import CollectAgent

OUT_DIR = os.path.join("results", "collect")
_write_lock = threading.Lock()


def run_one(origin_split: str, task_id: int, trial: int, run_id: str,
            max_num_steps: int, split_label: str):
    """One trajectory. Returns (rows, reward, n_messages)."""
    from tau_bench.envs.retail.env import MockRetailDomainEnv

    env = MockRetailDomainEnv(
        user_strategy="llm",
        user_model=config.USER_MODEL,
        user_provider=config.PROVIDER,
        task_split=origin_split,
        task_index=task_id,
    )
    agent = CollectAgent(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model=config.AGENT_MODEL,
        provider=config.PROVIDER,
    )
    agent.split_label = split_label
    agent.trial = trial
    result, records = agent.solve(
        env, task_index=task_id, max_num_steps=max_num_steps
    )
    rows = []
    for r in records:
        row = dataclasses.asdict(r)
        row.update(
            origin_split=origin_split,
            reward=result.reward,
            run_id=run_id,
            agent_model=config.AGENT_MODEL,
            user_model=config.USER_MODEL,
        )
        rows.append(row)
    return rows, result.reward, len(result.messages)


def task_list(split: str):
    """(origin_split, task_id, split_label) triples for the given block."""
    fit_ids, cal_train_ids = config.fit_cal_ids()
    out = []
    if split in ("fit", "all"):
        out += [("train", i, "fit") for i in sorted(fit_ids)]
    if split in ("cal", "all"):
        out += [("train", i, "cal") for i in sorted(cal_train_ids)]
        out += [("dev", i, "cal") for i in range(20)]
    return out


def done_keys(path: str):
    """Triples of completed trajectories. Step rows AND traj markers count,
    so a trajectory with zero gated actions (no step rows) is still done."""
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["origin_split"], r["task_id"], r["trial"]))
                except Exception:
                    pass
    return done


def _append_rows(path: str, rows):
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_with_retry(origin_split, task_id, trial, run_id, max_num_steps,
                   split_label, attempts=2):
    last = None
    for _ in range(attempts):
        try:
            return run_one(origin_split, task_id, trial, run_id,
                           max_num_steps, split_label)
        except llm.BudgetExceeded:
            raise
        except Exception as e:  # rate limits, transient API errors
            last = e
            time.sleep(20)
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["fit", "cal", "all"], default="all")
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tasks", type=int, default=0, help="limit tasks (smoke)")
    ap.add_argument("--max-num-steps", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "rows.jsonl"))
    args = ap.parse_args()

    llm.install()
    unpriced = [m for m in (config.AGENT_MODEL, config.USER_MODEL) if m not in config.PRICES]
    if unpriced and os.environ.get("GATE_ALLOW_UNPRICED") != "1":
        print(
            f"ERROR: no price configured for {unpriced}; add it to "
            "gate/config.py PRICES or set GATE_ALLOW_UNPRICED=1 (disables "
            "the USD guard for those calls).",
            flush=True,
        )
        sys.exit(2)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    manifest_path = os.path.join(
        os.path.dirname(args.out), "manifest_" + run_id + ".json"
    )
    manifest = {
        "run_id": run_id,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": sys.argv[1:],
        "agent_model": config.AGENT_MODEL,
        "user_model": config.USER_MODEL,
        "provider": config.PROVIDER,
        "split": args.split,
        "trials": args.trials,
        "split_seed": config.SPLIT_SEED,
        "fit_size": config.FIT_SIZE,
        "agreement_k": config.AGREEMENT_K,
        "agreement_temperature": config.AGREEMENT_TEMPERATURE,
        "prices_checked": "2026-10-13",
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    done = done_keys(args.out)
    jobs = []
    for origin, tid, label in task_list(args.split):
        for trial in range(args.trials):
            if (origin, tid, trial) not in done:
                jobs.append((origin, tid, trial, label))
    if args.max_tasks:
        jobs = jobs[: args.max_tasks]
    print(f"{len(jobs)} trajectories to run (skipped {len(done)//max(args.trials,1)} done)", flush=True)

    n_rows = 0
    n_traj = 0
    errors = []
    stop = False
    handled = set()
    t0 = time.time()

    def handle(fut, key):
        """Write rows + marker, print progress. Returns True on success."""
        nonlocal n_rows, n_traj
        o, t, tr = key
        rows, reward, n_msgs = fut.result()
        marker = {
            "kind": "traj_done",
            "origin_split": o,
            "task_id": t,
            "trial": tr,
            "reward": reward,
            "n_records": len(rows),
            "n_messages": n_msgs,
            "run_id": run_id,
        }
        _append_rows(args.out, rows + [marker])
        n_rows += len(rows)
        n_traj += 1
        snap = llm.tracker.snapshot()
        print(
            f"[{len(errors)}err] {o}/{t}/tr{tr} reward={reward:.0f} "
            f"gated={len(rows)} msgs={n_msgs} usd={snap['_total_usd']:.2f} "
            f"({time.time()-t0:.0f}s)",
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {
            ex.submit(run_with_retry, o, t, tr, run_id, args.max_num_steps, lab): (o, t, tr)
            for o, t, tr, lab in jobs
        }
        for fut in as_completed(futs):
            key = futs[fut]
            handled.add(fut)
            try:
                handle(fut, key)
            except llm.BudgetExceeded as e:
                print(f"BUDGET STOP: {e}", flush=True)
                stop = True
                for other in futs:
                    other.cancel()
                break
            except Exception as e:
                errors.append({"job": list(key), "error": repr(e),
                               "tb": traceback.format_exc()[-2000:]})
                print(f"ERROR {key[0]}/{key[1]}/tr{key[2]}: {e!r}", flush=True)
        if stop:
            # drain trajectories that already finished during shutdown:
            # their rows are paid for, keep them
            for fut, key in futs.items():
                if fut not in handled and fut.done() and not fut.cancelled():
                    try:
                        handle(fut, key)
                    except Exception:
                        pass

    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["trajectories_run"] = n_traj
    manifest["rows_written"] = n_rows
    manifest["errors"] = errors
    manifest["usage"] = llm.tracker.snapshot()
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest["usage"], indent=2), flush=True)
    print(f"manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
