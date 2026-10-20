#!/usr/bin/env python3
"""Run the complete QCAR audit workflow in the required order.

Stages (each script reads the outputs of the stages before it):

  0  01_extract_data.py                  optional: rebuild data/*.csv from the project documents
  1  02_audit_dataset.py                 S1/S2 base audit: closure, ordering, PCA, shift, source distances
  2  03_benchmark_models.py              S3/S4: 7 regressors x 3 protocols, stored ANN, order-only trend
  3  04_interpretability_uncertainty.py  S5/S6: permutation importance, split-conformal (Extra Trees)
  4  07_baseline_ann_replica.py          S3: Levenberg-Marquardt replica of the 9-9-1 baseline ANN
  5  06_extended_audit.py                S1-S6 extensions (needs stages 1-4)
  6  08_make_figures_v2.py               Figures 2-14 (needs pdflatex)
  7  10_graphical_abstract.py            Figure 1 / graphical abstract (needs pdflatex)
  8  09_make_tables_v2.py                Tables 8-16 (LaTeX)

By default a stage is skipped when its outputs already exist; use --force to
recompute everything from data/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RESULTS = PROJECT_ROOT / "results"
FIGURES = PROJECT_ROOT / "figures"
TABLES = PROJECT_ROOT / "tables"


def run_stage(label: str, command: list[str]) -> None:
    print(f"\n[{label}] {' '.join(command)}", flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print(f"[{label}] completed in {time.perf_counter() - started:.1f} s", flush=True)


def run_if_needed(label: str, script: str, expected: list[Path], force: bool) -> None:
    if not force and expected and all(p.exists() for p in expected):
        print(f"[{label}] skipped; outputs already exist (use --force to recompute).")
        return
    run_stage(label, [sys.executable, str(SCRIPTS_DIR / script)])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce every result, figure and table of the paper.")
    p.add_argument("--force", action="store_true", help="Recompute all outputs even if they exist.")
    p.add_argument("--skip-figures", action="store_true",
                   help="Skip the figure stages (use when no LaTeX installation is available).")
    p.add_argument("--extract", action="store_true",
                   help="Re-extract data/*.csv from the original project documents first.")
    p.add_argument("--final-report", type=Path)
    p.add_argument("--lookup-docx", type=Path)
    p.add_argument("--lookup-xlsx", type=Path)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    if a.extract:
        if not (a.final_report and a.lookup_docx and a.lookup_xlsx):
            raise SystemExit("--extract needs --final-report, --lookup-docx and --lookup-xlsx.")
        run_stage("0 extraction", [sys.executable, str(SCRIPTS_DIR / "01_extract_data.py"),
                                   "--final-report", str(a.final_report),
                                   "--lookup-docx", str(a.lookup_docx),
                                   "--lookup-xlsx", str(a.lookup_xlsx)])
    if not (PROJECT_ROOT / "data" / "qcar_667.csv").exists():
        raise SystemExit("data/qcar_667.csv is missing.")

    run_if_needed("1 base audit", "02_audit_dataset.py", [RESULTS / "audit_summary.json"], a.force)
    run_if_needed("2 benchmark", "03_benchmark_models.py",
                  [RESULTS / "model_benchmark_summary.csv", RESULTS / "order_trend_metrics.csv"], a.force)
    run_if_needed("3 attribution and conformal", "04_interpretability_uncertainty.py",
                  [RESULTS / "importance_stability.csv", RESULTS / "conformal_summary.csv"], a.force)
    run_if_needed("4 baseline ANN replica", "07_baseline_ann_replica.py",
                  [RESULTS / "ann_replica_summary.json"], a.force)
    run_if_needed("5 extended audit S1-S6", "06_extended_audit.py",
                  [RESULTS / "negative_controls.csv", RESULTS / "conformal_extended.csv"], a.force)
    if a.skip_figures:
        print("\n[6-7 figures] skipped (--skip-figures); pre-built figures are in figures/.")
    else:
        run_if_needed("6 figures 2-14", "08_make_figures_v2.py", [FIGURES / "fig_s6_conformal.pdf"], a.force)
        run_if_needed("7 graphical abstract", "10_graphical_abstract.py",
                      [FIGURES / "fig_graphical_abstract.pdf"], a.force)
    run_if_needed("8 tables 8-16", "09_make_tables_v2.py", [TABLES / "tab_res_conformal.tex"], a.force)
    print(f"\nWorkflow complete.\nResults: {RESULTS}\nFigures: {FIGURES}\nTables:  {TABLES}")


if __name__ == "__main__":
    main()
