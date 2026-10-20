# QCAR audit suite for Al–Li alloy composition–property data

Code and data for the article

> S. Sarkar and P. Chakraborty, *Audit of Quantitative Composition Activity in Aluminium-Lithium (Al-Li) Alloy Property Data using Machine Learning*, submitted to *Computational Materials Science* (2026).

The repository reproduces every number, table and figure of the article from the
extracted data in `data/`. All intermediate results are also included, so each
value in the paper can be checked without re-running anything.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts/run_all.py --force    # recompute everything, in the order below
```

Without `--force`, stages whose outputs already exist are skipped. Add
`--skip-figures` if no LaTeX installation is available; the pre-built figures in
`figures/` are then kept.

## Run order

The scripts must run in this order, because each stage reads the outputs of the
stages above it. `scripts/run_all.py` runs them in this order automatically.

| Step | Script | Audit strategy | What it does | Main outputs (`results/`) |
|---|---|---|---|---|
| 0 (optional) | `01_extract_data.py` | – | Rebuilds `data/*.csv` from the original project documents (not distributed; see below) | `data/*.csv` |
| 1 | `02_audit_dataset.py` | S1, S2 | Closure, uniqueness, ordering, PCA, train–test shift, nearest-source distances | `audit_summary.json`, `order_correlations.csv`, `split_shift.csv`, `source_anchor_distances.csv` |
| 2 | `03_benchmark_models.py` | S3, S4 | 7 regressors × 3 validation protocols, stored baseline ANN scoring, order-only trend | `model_benchmark_*.csv`, `model_predictions.csv`, `order_trend_*.csv` |
| 3 | `04_interpretability_uncertainty.py` | S5, S6 | Permutation importance and split-conformal intervals (Extra Trees) | `permutation_importance_*.csv`, `importance_stability.csv`, `conformal_*.csv` |
| 4 | `07_baseline_ann_replica.py` | S3 | Levenberg–Marquardt replica of the 9–9–1 baseline ANN under all protocols (uses `ann_replica.py`) | `ann_replica_*.csv`, `ann_replica_summary.json` |
| 5 | `06_extended_audit.py` | S1–S6 | Physical admissibility, record linkage, contamination, scale sensitivity, block bootstrap, error–distance, negative controls, grouped importance, extra conformal models | `physics_*.csv`, `tensile_record_linkage.csv`, `negative_controls.csv`, `conformal_extended.csv`, `extended_summary.json`, … |
| 6 | `08_make_figures_v2.py` | – | Figures 2–14 | `figures/fig_*.{pdf,pgf,png}` |
| 7 | `10_graphical_abstract.py` | – | Figure 1 (graphical abstract) | `figures/fig_graphical_abstract.*` |
| 8 | `09_make_tables_v2.py` | – | Tables 8–16 as LaTeX | `tables/tab_res_*.tex` |

`common.py` holds the shared paths, column names, model definitions and the random seed (20260806).

## Where each figure and table comes from

| Paper item | File | Produced by |
|---|---|---|
| Fig. 1 (graphical abstract) | `figures/fig_graphical_abstract.*` | `10_graphical_abstract.py` |
| Fig. 2 Ordering of the master table | `figures/fig_s1_sequences.*` | `08_make_figures_v2.py` |
| Fig. 3 Physical admissibility | `figures/fig_s1_physics.*` | `08_make_figures_v2.py` |
| Fig. 4 Source fidelity and record linkage | `figures/fig_s1_provenance.*` | `08_make_figures_v2.py` |
| Fig. 5 PCA | `figures/fig_s2_pca.*` | `08_make_figures_v2.py` |
| Fig. 6 Nearest-neighbour distances | `figures/fig_s2_nn.*` | `08_make_figures_v2.py` |
| Fig. 7 Validation-regime benchmark | `figures/fig_s3_benchmark.*` | `08_make_figures_v2.py` |
| Fig. 8 Contiguous blocks; error vs distance | `figures/fig_s3_blocks_distance.*` | `08_make_figures_v2.py` |
| Fig. 9 Baseline ANN replica | `figures/fig_s3_ann.*` | `08_make_figures_v2.py` |
| Fig. 10 Forward-holdout predictions | `figures/fig_s3_forward.*` | `08_make_figures_v2.py` |
| Fig. 11 Negative controls | `figures/fig_s4_controls.*` | `08_make_figures_v2.py` |
| Fig. 12 Feature attribution | `figures/fig_s5_attribution.*` | `08_make_figures_v2.py` |
| Fig. 13 Split-conformal intervals | `figures/fig_s6_conformal.*` | `08_make_figures_v2.py` |
| Fig. 14 Preliminary network | `figures/fig_a1_preliminary.*` | `08_make_figures_v2.py` |
| Table 5 Data dictionary | `results/support_ranges.csv` | `06_extended_audit.py` |
| Table 8 Integrity and provenance | `tables/tab_res_audit.tex` | `09_make_tables_v2.py` |
| Table 9 Univariate shift | `tables/tab_res_shift.tex` | `09_make_tables_v2.py` |
| Table 10 Nearest-neighbour distance | `tables/tab_res_nn.tex` | `09_make_tables_v2.py` |
| Table 11 Pooled R² and nMAE | `tables/tab_res_benchmark.tex` | `09_make_tables_v2.py` |
| Table 12 Baseline ANN evidence | `tables/tab_res_ann.tex` | `09_make_tables_v2.py` |
| Table 13 Forward holdout with bootstrap intervals | `tables/tab_res_forward.tex` | `09_make_tables_v2.py` |
| Table 14 Order-only trend and negative controls | `tables/tab_res_controls.tex` | `09_make_tables_v2.py` |
| Table 15 Attribution stability | `tables/tab_res_attribution.tex` | `09_make_tables_v2.py` |
| Table 16 Conformal intervals | `tables/tab_res_conformal.tex` | `09_make_tables_v2.py` |

Each figure is saved as `.pdf` (text typeset by pdflatex), `.pgf` (for `\input` in LaTeX) and `.png` (preview).

## Data (`data/`)

| File | Content |
|---|---|
| `qcar_667.csv` | Master QCAR table: `row_id`, stored partition label (`supplied_split`: rows 0–498 `training`, 499–666 `testing`), nine mass fractions in wt.% (Al, Cu, Li, Mg, Mn, Zr, Si, Zn, Cd), four properties (tensile strength, proof stress and shear strength in MPa; Vickers hardness in HV), stored baseline ANN predictions and stored squared relative errors |
| `source_lookup_compositions.csv` | 94-record source compilation (blank cells read as 0) with tensile strength and the preliminary network's predictions |
| `source_lookup_four_properties.csv` | 27-record four-property source table |
| `source_prediction_table.csv` | Recorded and ANN-predicted four properties for 26 records |
| `handbook_temper_sheet.csv` | 52 alloy–temper tensile strengths (ksi, MPa) of the handbook sheet |
| `preliminary_network_scaled.csv` | Scaled predictions and targets of the preliminary network (63 records) |
| `lookup_xlsx_*_block.csv` | Raw spreadsheet blocks, kept for traceability |

The original project documents (Word and Excel files) are not distributed; they are
available from the corresponding author on reasonable request. Step 0 is therefore
optional and all analyses start from `data/`.

## Requirements

- Python 3.12 (3.11 or later is required by the pinned packages); packages in `requirements.txt`.
- Figures (steps 6–7): a LaTeX installation with `pdflatex`, `amsmath` and `amssymb`, and
  `pdftoppm` (poppler) for the PNG previews. Use `--skip-figures` otherwise.
- Run time: about 7 minutes on a single CPU core (step 5 is the longest, about 3 minutes).

All stochastic steps use the fixed seed 20260806. With other library versions,
results may differ in the last reported digit.

## Licence

Code: MIT (see `LICENSE`). Data and results: CC BY 4.0.

## Citation

See `CITATION.cff`. Please cite the article once published.

## Contact

Priyam Chakraborty, Department of Aerospace Engineering, Indian Institute of Technology Kharagpur — priyam@aero.iitkgp.ac.in
