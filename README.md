# Protein Structure Analysis Project

## Project Overview
This repository contains various implementations and experiments for protein structure analysis using different deep learning approaches.

## Reproducible Pipeline (Binding Site Prediction)

The `run_pipeline.py` script provides an end-to-end, reproducible pipeline extracted from `GraphSAGE-improving.ipynb`. It supports multiple GNN backbones: **GraphSAGE**, **GCN**, and **GAT**.

### Quick Start

```bash
# Full run with default config (GraphSAGE)
python run_pipeline.py --config configs/graphsage_default.yaml

# Select backbone via CLI
python run_pipeline.py --config configs/graphsage_default.yaml --model gcn
python run_pipeline.py --config configs/graphsage_default.yaml --model gat

# Smoke test (minimal data for quick verification)
python run_pipeline.py --config configs/graphsage_default.yaml --smoke
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--config` | Path to YAML config (default: built-in defaults) |
| `--model` | GNN backbone: `graphsage`, `gcn`, `gat` |
| `--device` | Device: `cuda` or `cpu` |
| `--seed` | Random seed for reproducibility |
| `--save-dir` | Output directory for checkpoints and metrics |
| `--smoke` | Smoke test with 4 train + 2 test samples |

### Data Paths

Update `configs/graphsage_default.yaml` or pass a custom config:

- `train_csv`: Training CSV with `prot_id`, `sequence`, `labels` (list format)
- `test_csv`: Test CSV with same columns
- `pdb_dir`: Directory containing PDB files (e.g. `{prot_id}.pdb` or `{prot_id}_alphafold.pdb`)

### Artifacts

Outputs are saved under `artifacts/` (or `--save-dir`):

- `{backbone}_best_model.pth`: Best model checkpoint
- `run_metadata.json`: Config, metrics, timestamp

### Pipeline Layout

- `pipeline/config.py` – Configuration dataclasses
- `pipeline/io.py` – Data loading and path resolution
- `pipeline/embeddings.py` – ESM-2 tokenization and embeddings
- `pipeline/graph_features.py` – Structure features and graph construction
- `pipeline/models.py` – GCN, GraphSAGE, GAT backbones
- `pipeline/losses.py` – Loss functions and binding features
- `pipeline/train.py` – Training loop
- `pipeline/evaluate.py` – Evaluation metrics

Existing helper scripts (`data_preparation.py`, `features_extraction.py`, `alphafold_data_ingestion.py`, etc.) remain for creating multi-label and processed datasets.

## Repository Structure
- `run_pipeline.py`: Main CLI for binding site prediction
- `pipeline/`: Modular pipeline package
- `configs/`: YAML configuration files
- `GraphSAGE-improving.ipynb`: Original notebook (reference)
- `data_preparation.py`, `features_extraction.py`, etc.: Dataset preparation helpers

## Important Notice

Before running any code:

1. **Data Paths**: Update paths in `configs/graphsage_default.yaml` or your config
2. **PDB Files**: Ensure `data/esmFold_pdb_files` (or your `pdb_dir`) contains PDB files for all protein IDs in train/test CSVs
3. **Execution**: `run_pipeline.py` handles embeddings, graph construction, training, and evaluation in one command

## Prerequisites
- Python 3.9+
- Required packages: `pip install -r requirements.txt`
- PyTorch, torch-geometric, transformers, mdtraj, pykan

## Contact
For questions or issues, please open a GitHub issue in this repository.