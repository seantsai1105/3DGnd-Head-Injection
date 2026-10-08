"""Summarize LIBERO / LIBERO-PRO evaluation results next to the paper (GR00T-N1.6 + Ours).

Usage:
    python examples/LIBERO/summarize_results.py <output_dir or results.jsonl> [...] [--per_task]

Each input is a results.jsonl written by gr00t/eval/rollout_policy.py (one line per task); several
inputs are shown side by side. The success rate of a suite is the mean over its tasks (10 tasks x
50 episodes in the paper). The paper column is Table 1.
"""

import argparse
import json
from pathlib import Path
import sys


TASKS_PER_SUITE = 10
SETTINGS = ("n_episodes", "n_envs", "max_episode_steps", "n_action_steps")
# (group, suite, label, paper success rate in %: Table 1)
SUITES = (
    ("In distribution", "libero_object", "LIBERO-Object", 99.6),
    ("In distribution", "libero_spatial", "LIBERO-Spatial", 92.4),
    ("Task perturbation", "libero_pro_object_task", "LIBERO-PRO Object", 70.4),
    ("Task perturbation", "libero_pro_spatial_task", "LIBERO-PRO Spatial", 84.6),
    ("Position perturbation", "libero_pro_object_swap", "LIBERO-PRO Object", 79.4),
    ("Position perturbation", "libero_pro_spatial_swap", "LIBERO-PRO Spatial", 41.0),
)


def load_results(path):
    """Return the results file and {env_name: record}; a task's last line overrides earlier ones."""
    path = Path(path)
    if path.is_dir():
        path = path / "results.jsonl"
    records = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if record["env_name"] in records:
                    print(f"{path}: using the last result of {record['env_name']}", file=sys.stderr)
                records[record["env_name"]] = record
    return path, records


def suite_rate(records, suite):
    """Mean success rate (%) over the evaluated tasks of a suite, and the number of tasks."""
    rates = [r["success_rate"] for r in records.values() if r["suite"] == suite]
    return (100.0 * sum(rates) / len(rates) if rates else None), len(rates)


def format_rate(rate, n_tasks):
    if rate is None:
        return "-"
    if n_tasks != TASKS_PER_SUITE:
        return f"{rate:.1f} ({n_tasks}/{TASKS_PER_SUITE})"
    return f"{rate:.1f}"


def format_average(suite_rates):
    """Average over suites; '*' marks an average that includes incomplete suites."""
    rates = [rate for rate, _ in suite_rates]
    if None in rates:
        return "-"
    complete = all(n_tasks == TASKS_PER_SUITE for _, n_tasks in suite_rates)
    return f"{sum(rates) / len(rates):.1f}" + ("" if complete else "*")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("inputs", nargs="+", help="output_dir(s) or results.jsonl file(s)")
    parser.add_argument("--per_task", action="store_true", help="also list every task")
    args = parser.parse_args()

    loaded = [load_results(p) for p in args.inputs]
    for i, (path, records) in enumerate(loaded, 1):
        n_episodes = sum(r["n_episodes"] for r in records.values())
        print(f"[{i}] {path}: {len(records)} tasks, {n_episodes} episodes")
        settings = {tuple(r.get(key) for key in SETTINGS) for r in records.values()}
        for values in sorted(settings, key=str):
            print("    " + ", ".join(f"{key} {value}" for key, value in zip(SETTINGS, values)))
        if len(settings) > 1:
            print(f"WARNING: {path} mixes results of different settings", file=sys.stderr)

    paper = {suite: rate for _, suite, _, rate in SUITES}
    columns = "".join(f"{f'[{i}]':>14s}" for i in range(1, len(loaded) + 1))
    print(f"\n{'':22s}{'Success rate (%)':20s}{columns}{'Paper':>8s}")
    for group in dict.fromkeys(group for group, *_ in SUITES):
        suites = [(suite, label) for g, suite, label, _ in SUITES if g == group]
        for k, (suite, label) in enumerate(suites):
            cells = "".join(
                f"{format_rate(*suite_rate(records, suite)):>14s}" for _, records in loaded
            )
            print(f"{group if k == 0 else '':22s}{label:20s}{cells}{paper[suite]:>8.1f}")
        cells = "".join(
            f"{format_average([suite_rate(records, s) for s, _ in suites]):>14s}"
            for _, records in loaded
        )
        paper_average = sum(paper[suite] for suite, _ in suites) / len(suites)
        print(f"{'':22s}{'Avg.':20s}{cells}{paper_average:>8.1f}")
    print("(n/10): number of evaluated tasks of an incomplete suite; *: includes incomplete suites")

    if args.per_task:
        for _, suite, _, _ in SUITES:
            names = sorted({n for _, rec in loaded for n, r in rec.items() if r["suite"] == suite})
            if not names:
                continue
            print(f"\n{suite}")
            for name in names:
                cells = "".join(
                    f"{100 * records[name]['success_rate']:>8.1f}"
                    if name in records
                    else f"{'-':>8s}"
                    for _, records in loaded
                )
                print(f"  {name.removeprefix('libero_sim/'):110s}{cells}")


if __name__ == "__main__":
    main()
