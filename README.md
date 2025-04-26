# Protein Structure Analysis Project

## Project Overview
This repository contains various implementations and experiments for protein structure analysis using different deep learning approaches.

## Repository Structure
- `GAT-experiments-23Apr`: Final implementation of ESM-2 + GAT with/without KAN
- `GAT-GCN-improved-20APr`: Final implementation of ESM-2 + GCN with/without KAN
- `protein_folding`: Code for protein structure prediction using ESMFold
- `esm-3-emebddings-generation`: ESM-3 embeddings generation via API calls
- `ESM-3_and_GCN`: Experimental code combining ESM-3 with GCN

## Important Notice ⚠️

Before running any code in this repository:

1. **Data Paths**: 
    - Modify all data file paths according to your local setup
    - Data we used for training and testing: `data/development_set/full_grouped_train_binding_sites_df.csv` (Training data), `data/development_set/full_grouped_test_binding_sites_df.csv (TestSet300)`, `data/independent_set/grouped_test_46_new_binding_sites.csv` (TestSetNew46)
    - Data file path for storing protein's predicted structures using ESMFold: `data/esmFold_pdb_files`
    - Check and update paths in configuration files to align with your local file path.
    - Ensure data files are in the correct locations

2. **Execution Order**:
    - Generate embeddings first using the ESM-3 code
    - Run model training scripts afterward
    - Follow each directory's specific README for detailed instructions

## Prerequisites
- Python 3.9+
- Required packages (refer to requirements.txt)
- Access to ESM-3 API (for embeddings generation)

## Contact
For questions or issues, please open a GitHub issue in this repository.