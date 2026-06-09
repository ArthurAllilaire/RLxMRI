"""Plot held-out full-Bloch MAPE vs cumulative wallclock for MF curriculum runs.

The money-plot for report/e2_runs/multi_fidelity.md: compares the criterion
curriculum against the fixed-schedule and full-Bloch-only baselines on the axis
that matters — **wallclock seconds**, since the fidelities cost wildly different
amounts per step. Switch points are drawn as vertical lines annotated with the
fidelity gap.

Usage:
    python python/plot_mf_curriculum.py \
        runs/e2/mf_criterion:Criterion \
        runs/e2/mf_fixed:Fixed-schedule \
        runs/e2/full_only:Full-only \
        --out runs/e2/mf_moneyplot.png

Each arg is `dir[:label]`. An MF run is detected by fidelity_history.json; a
single-fidelity run falls back to eval_history.json (its in-fidelity == full).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _stage0_wall_origin(entries: list[dict]) -> float:
    """Wallclock origin for MF histories.

    train_e2_mf.py records `wall_s` as absolute Unix epoch seconds. Older
    histories do not emit a stage_start row for stage 0, so the stage-0 start is
    represented by the earliest timestamp in the file. If future histories add
    an explicit stage-0 start event, it will also be the earliest timestamp.
    """
    walls = [float(e["wall_s"]) for e in entries if "wall_s" in e]
    return min(walls) if walls else 0.0


def _load_curve(run_dir: Path):
    """Return (wall_rel, mape_full, switches) for a run.

    wall_rel    : wallclock seconds since the run started
    mape_full   : held-out full-Bloch MAPE (%) at each sample
    switches    : list of (wall_rel, gap_pct) for stage boundaries (MF runs only)
    """
    fh = run_dir / "fidelity_history.json"
    if fh.exists():
        entries = json.loads(fh.read_text())
        t0 = _stage0_wall_origin(entries)
        xs, ys, switches = [], [], []
        for e in entries:
            w = e.get("wall_s", t0) - t0
            if e.get("kind") == "decision" and "mape_H_pct" in e:
                xs.append(w); ys.append(e["mape_H_pct"])
            elif e.get("kind") == "stage_end" and "full_mape_pct" in e:
                xs.append(w); ys.append(e["full_mape_pct"])
            elif e.get("kind") == "stage_start":
                switches.append((w, e.get("fidelity_gap_pct")))
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        return [xs[i] for i in order], [ys[i] for i in order], switches

    # single-fidelity fallback: eval_history.json (in-fidelity == full sim)
    eh = run_dir / "eval_history.json"
    if eh.exists():
        entries = json.loads(eh.read_text())
        xs = [e.get("wall_s", 0.0) for e in entries]
        ys = [e["mape_pct"] for e in entries]
        return xs, ys, []
    raise FileNotFoundError(f"no fidelity_history.json or eval_history.json in {run_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="dir[:label] for each run to overlay")
    p.add_argument("--out", type=Path, default=Path("mf_moneyplot.png"))
    p.add_argument("--title", default="Multi-fidelity curriculum: full-sim MAPE vs wallclock")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5))
    for spec in args.runs:
        run_dir, _, label = spec.partition(":")
        run_dir = Path(run_dir)
        label = label or run_dir.name
        xs, ys, switches = _load_curve(run_dir)
        if not xs:
            print(f"[warn] no points for {run_dir}")
            continue
        line, = ax.plot([x / 3600 for x in xs], ys, marker="o", ms=3, label=label)
        for w, gap in switches:
            ax.axvline(w / 3600, color=line.get_color(), ls="--", alpha=0.4)
            if gap is not None:
                ax.annotate(f"gap {gap:+.1f}%", (w / 3600, max(ys)),
                            color=line.get_color(), fontsize=8, rotation=90,
                            va="top", ha="right")

    ax.set_xlabel("cumulative wallclock (hours)")
    ax.set_ylabel("held-out full-Bloch MAPE (%)")
    ax.set_yscale("log")
    ax.set_title(args.title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
