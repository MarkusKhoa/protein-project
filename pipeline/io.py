"""Data I/O: CSV loading, label parsing, path normalization."""
import ast
from pathlib import Path
from typing import Optional

import pandas as pd


def load_train_test(
    train_csv: str,
    test_csv: str,
    label_column: str = "labels",
    excluded_protein_ids: Optional[list] = None,
    id_column: str = "id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test CSV with label parsing.

    Expects labels as string representation of Python list, e.g. "[1, 0, 1, 0]".
    """
    excluded_protein_ids = excluded_protein_ids or []

    train_df = pd.read_csv(train_csv)
    train_df[label_column] = train_df[label_column].apply(ast.literal_eval)
    if id_column != "id" and id_column in train_df.columns:
        train_df.rename(columns={id_column: "id"}, inplace=True)
    if excluded_protein_ids:
        train_df = train_df[~train_df["id"].isin(excluded_protein_ids)]

    test_df = pd.read_csv(test_csv)
    test_df[label_column] = test_df[label_column].apply(ast.literal_eval)
    if id_column != "id" and id_column in test_df.columns:
        test_df.rename(columns={id_column: "id"}, inplace=True)

    return train_df, test_df


def resolve_pdb_path(prot_id: str, pdb_dir: str) -> Path:
    """Resolve PDB file path. Tries {prot_id}.pdb (ESMFold) first."""
    base = Path(pdb_dir)
    candidates = [
        base / f"{prot_id}.pdb",
        base / f"{prot_id}_alphafold.pdb",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No PDB found for {prot_id} in {pdb_dir}")
