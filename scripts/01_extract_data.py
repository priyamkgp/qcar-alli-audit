#!/usr/bin/env python3
"""Extract the supplied QCAR tables into tidy CSV files.

The source workbooks are not redistributed with this project. The already
extracted CSV files are included. This script allows the extraction to be
repeated when the original DOCX/XLSX files are available.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from docx import Document
from openpyxl import load_workbook

from common import DATA_DIR, FEATURES, RESULTS_DIR, TARGETS, ensure_directories


def table_to_frame(table) -> pd.DataFrame:
    rows: List[List[str]] = []
    for row in table.rows:
        rows.append([cell.text.strip() for cell in row.cells])
    if not rows:
        return pd.DataFrame()
    header = rows[0]
    # Word can duplicate merged-cell headers. Make column names unique.
    seen = {}
    unique_header = []
    for value in header:
        value = value.strip() or "unnamed"
        count = seen.get(value, 0)
        unique_header.append(value if count == 0 else f"{value}_{count+1}")
        seen[value] = count + 1
    return pd.DataFrame(rows[1:], columns=unique_header)


def coerce_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace("", np.nan), errors="coerce")
    return out


def extract_qcar(final_report: Path) -> pd.DataFrame:
    doc = Document(final_report)
    if len(doc.tables) < 3:
        raise ValueError("Expected at least three tables in the final report DOCX.")

    train = table_to_frame(doc.tables[1])
    test = table_to_frame(doc.tables[2])
    required = FEATURES + TARGETS
    for name, frame in (("training", train), ("testing", test)):
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing {missing} from the {name} table.")

    train = coerce_numeric(train, train.columns)
    test = coerce_numeric(test, test.columns)
    train = train.dropna(subset=required).copy()
    test = test.dropna(subset=required).copy()
    train.insert(0, "supplied_split", "training")
    test.insert(0, "supplied_split", "testing")
    qcar = pd.concat([train, test], ignore_index=True)
    qcar.insert(0, "row_id", np.arange(len(qcar), dtype=int))
    return qcar


def _clean_source_frame(frame: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    """Keep designation rows, read blank composition cells as zero.

    The lookup DOCX contains repeated header rows and five records with blank
    composition cells. Every such record closes to 100 wt.% when its blanks are
    read as zero, so blanks are treated as zero rather than as missing.
    """
    out = frame.copy()
    first = out.columns[0]
    out = out.rename(columns={first: "alloy"})
    out = out[out["alloy"].astype(str).str.strip().str.fullmatch(r"\d{4}")].copy()
    out = coerce_numeric(out, FEATURES + value_columns)
    out[FEATURES] = out[FEATURES].fillna(0.0)
    out["alloy"] = out["alloy"].astype(int)
    out.insert(0, "record", np.arange(len(out), dtype=int))
    return out.reset_index(drop=True)


def extract_lookup_docx(lookup_docx: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    doc = Document(lookup_docx)
    if len(doc.tables) < 4:
        raise ValueError("Expected four tables in the lookup DOCX.")

    tensile_lookup = table_to_frame(doc.tables[1])
    tensile_lookup = _clean_source_frame(
        tensile_lookup,
        [
            "Original Tensile Strength",
            "Predict value in  50000 iteration",
            "Predict value in 100000  iteration",
        ],
    )
    tensile_lookup = tensile_lookup.rename(
        columns={
            "Original Tensile Strength": "tensile_strength",
            "Predict value in  50000 iteration": "prelim_pred_50k",
            "Predict value in 100000  iteration": "prelim_pred_100k",
        }
    )
    keep = ["record", "alloy"] + FEATURES + ["tensile_strength", "prelim_pred_50k", "prelim_pred_100k"]
    tensile_lookup = tensile_lookup[keep]

    four_property_lookup = table_to_frame(doc.tables[2])
    four_property_lookup = _clean_source_frame(
        four_property_lookup,
        ["Tensile Strenth", "Proof Stress (mpa)", "Shear Strength (mpa)", "Hardness Vickers"],
    ).rename(
        columns={
            "Tensile Strenth": "tensile_strength",
            "Proof Stress (mpa)": "proof_stress",
            "Shear Strength (mpa)": "shear_strength",
            "Hardness Vickers": "hardness",
        }
    )
    four_property_lookup = four_property_lookup[
        ["record", "alloy"] + FEATURES + ["tensile_strength", "proof_stress", "shear_strength", "hardness"]
    ]

    # Prediction table: predicted (cols 1-4) and recorded "original" (cols 5-8) values.
    rows = []
    for row in doc.tables[3].rows:
        cells = [cell.text.strip() for cell in row.cells]
        if cells and cells[0].isdigit() and len(cells[0]) == 4:
            values = pd.to_numeric(pd.Series(cells[1:9]), errors="coerce").to_list()
            rows.append([int(cells[0])] + values)
    prediction_table = pd.DataFrame(
        rows,
        columns=[
            "alloy",
            "pred_tensile_strength", "pred_proof_stress", "pred_shear_strength", "pred_hardness",
            "orig_tensile_strength", "orig_proof_stress", "orig_shear_strength", "orig_hardness",
        ],
    )
    return tensile_lookup, four_property_lookup, prediction_table


def extract_handbook_sheet(xlsx_path: Path) -> pd.DataFrame:
    """Temper-resolved typical tensile strengths of the 11 handbook alloys (Sheet1)."""
    workbook = load_workbook(xlsx_path, data_only=True, read_only=True)
    worksheet = workbook["Sheet1"]
    rows, current = [], None
    for values in worksheet.iter_rows(min_row=7, values_only=True):
        alloy, temper, ksi, mpa = values[1], values[13], values[14], values[15]
        if alloy is not None:
            current = alloy
        if mpa is not None:
            rows.append({"alloy": int(current), "temper": str(temper), "ksi": float(ksi), "mpa": float(mpa)})
    frame = pd.DataFrame(rows)
    frame.insert(0, "position", np.arange(len(frame), dtype=int))
    return frame


def extract_preliminary_network(xlsx_path: Path) -> pd.DataFrame:
    """Scaled predictions (column N) and scaled targets (column O) on Sheet3."""
    workbook = load_workbook(xlsx_path, data_only=True, read_only=True)
    worksheet = workbook["Sheet3"]
    rows = []
    for values in worksheet.iter_rows(min_row=1, values_only=True):
        if len(values) > 14 and values[13] is not None and values[14] is not None:
            rows.append({"scaled_prediction": float(values[13]), "scaled_target": float(values[14])})
    frame = pd.DataFrame(rows)
    frame.insert(0, "sheet2_row", np.arange(len(frame), dtype=int))
    return frame


def extract_xlsx_blocks(xlsx_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract the two semantic blocks from Sheet2 of the supplied workbook.

    Rows 2--12 use the visible generic-alloy header. Rows 13 onward reuse those
    columns for the QCAR Al--Li composition semantics documented in the DOCX.
    """
    workbook = load_workbook(xlsx_path, data_only=True, read_only=True)
    worksheet = workbook["Sheet2"]

    generic_headers = [worksheet.cell(1, c).value for c in range(1, 11)]
    generic_rows = [
        [worksheet.cell(r, c).value for c in range(1, 11)] for r in range(2, 13)
    ]
    generic = pd.DataFrame(generic_rows, columns=generic_headers)

    qcar_headers = ["Cu", "Li", "Mg", "Mn", "Zr", "Si", "Zn", "Cd", "Al", "Tensile Strength"]
    qcar_rows = []
    for r in range(13, worksheet.max_row + 1):
        values = [worksheet.cell(r, c).value for c in range(1, 11)]
        if any(v is not None for v in values):
            qcar_rows.append(values)
    qcar_block = pd.DataFrame(qcar_rows, columns=qcar_headers)
    return generic, qcar_block


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final-report",
        type=Path,
        default=Path("/mnt/data/Final Report on Aluminium Alloy o QCAR MODEL - Copy.docx"),
    )
    parser.add_argument(
        "--lookup-docx",
        type=Path,
        default=Path("/mnt/data/Aluminium Alloy Lookup table and its details.docx"),
    )
    parser.add_argument(
        "--lookup-xlsx",
        type=Path,
        default=Path("/mnt/data/lookuptable.xlsx"),
    )
    args = parser.parse_args()
    ensure_directories()

    qcar = extract_qcar(args.final_report)
    tensile_lookup, four_property_lookup, prediction_table = extract_lookup_docx(args.lookup_docx)
    generic_xlsx, qcar_xlsx = extract_xlsx_blocks(args.lookup_xlsx)
    handbook = extract_handbook_sheet(args.lookup_xlsx)
    preliminary = extract_preliminary_network(args.lookup_xlsx)
    prediction_table.to_csv(DATA_DIR / "source_prediction_table.csv", index=False)
    handbook.to_csv(DATA_DIR / "handbook_temper_sheet.csv", index=False)
    preliminary.to_csv(DATA_DIR / "preliminary_network_scaled.csv", index=False)

    qcar.to_csv(DATA_DIR / "qcar_667.csv", index=False)
    tensile_lookup.to_csv(DATA_DIR / "source_lookup_compositions.csv", index=False)
    four_property_lookup.to_csv(
        DATA_DIR / "source_lookup_four_properties.csv", index=False
    )
    generic_xlsx.to_csv(DATA_DIR / "lookup_xlsx_generic_block.csv", index=False)
    qcar_xlsx.to_csv(DATA_DIR / "lookup_xlsx_qcar_block.csv", index=False)

    summary = {
        "qcar_rows": int(len(qcar)),
        "supplied_training_rows": int((qcar["supplied_split"] == "training").sum()),
        "supplied_testing_rows": int((qcar["supplied_split"] == "testing").sum()),
        "source_compilation_records": int(len(tensile_lookup)),
        "four_property_records": int(len(four_property_lookup)),
        "xlsx_generic_rows": int(len(generic_xlsx)),
        "xlsx_qcar_rows": int(len(qcar_xlsx)),
        "handbook_temper_entries": int(len(handbook)),
        "prediction_table_records": int(len(prediction_table)),
        "preliminary_network_rows": int(len(preliminary)),
    }
    (RESULTS_DIR / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
