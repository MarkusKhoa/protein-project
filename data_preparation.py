import ast
import torch.nn as nn
import pandas as pd
import numpy as np
import re

from Bio import SeqIO
from glob import glob
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          AutoModelForMaskedLM, DataCollatorForTokenClassification,
                          EsmForMaskedLM, EsmTokenizer,
                          TrainingArguments, Trainer)
from transformers.trainer_callback import ProgressCallback
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import train_test_split
from pprint import pprint
from datasets import Dataset
from datetime import datetime
from IPython.display import display
from tqdm import tqdm


def load_fasta_dataframe(file_path):
    """
    Load fasta file into a pandas dataframe
    :param file_path: path to fasta file
    :return: dataframe with columns 'id', 'sequence', 'seq_len'
    """
    # read fasta file
    records = [
        (record.id, str(record.seq)) for record in SeqIO.parse(file_path, "fasta")
    ]

    # create dataframe
    df_fasta = pd.DataFrame(records, columns=["id", "sequence"])
    df_fasta.rename(columns={"id": "prot_id"}, inplace=True)
    df_fasta["sequence_length"] = df_fasta["sequence"].apply(len)
    return df_fasta

def strimed_labels_string_to_list(s):
    for char in s:
        if char.isnumeric() or char == ",":
            pos = s.index(char)
            break
    protein_str = s[:pos].strip()
    labels_str = s[pos:].strip()
    labels_list = [int(label.strip()) for label in labels_str if label.strip().isdigit()]
    return protein_str, labels_list

def load_graphddips_fasta_dataframe(file_path):
    """
    Load fasta file into a pandas dataframe
    :param file_path: path to fasta file
    :return: dataframe with columns 'id', 'sequence', 'seq_len'
    """
    # read fasta file
    records = [
        (record.id, str(record.seq)) for record in SeqIO.parse(file_path, "fasta")
    ]

    protein_ids, proteins_lst, labels_lst = [], [], []

    for record in tqdm(records, total=len(records)):
        if not isinstance(record[1], str):
            raise ValueError(f"Invalid sequence for record {record[0]}: {record[1]}")
        protein_str, labels_list = strimed_labels_string_to_list(record[1])
        proteins_lst.append(protein_str)
        labels_lst.append(labels_list)
        protein_ids.append(record[0])
    
    initial_df = pd.DataFrame({
        "id": protein_ids,
        "sequence": proteins_lst,
        "labels": labels_lst
    })

    return initial_df

def load_binding_sites_dataframe(file_path, target=None):
    """
    Load binding sites file into a pandas dataframe
    :param file_path: path to binding sites file
    :param target: target protein class ('metal', 'nuclear', 'small')
    :return: dataframe with columns 'id', 'binding_sites'
    """
    # check if target is valid
    assert target in [
        "metal",
        "nuclear",
        "small",
    ], "target must be one of 'metal', 'nuclear', 'small'"

    # read binding sites file
    binding_sites = []
    with open(file_path, "r") as f:
        for line in f:
            protein_id, sites = line.strip().split("\t")
            binding_sites.append((protein_id, [int(site) for site in sites.split(",")]))

    # create dataframe
    df_binding_sites = pd.DataFrame(binding_sites, columns=["id", "binding_sites"])
    df_binding_sites["num_residues"] = df_binding_sites["binding_sites"].apply(len)
    df_binding_sites["target"] = target
    return df_binding_sites


def convert_to_binary_list(original_binding_sites_lst, sequence_len):
    """Convert a Binding-Active site string to a binary list based on the sequence length."""
    binary_list = [0] * sequence_len  # Initialize a list of zeros

    # Ensure original_binding_sites_lst is a list and not empty
    if isinstance(original_binding_sites_lst, list) and len(original_binding_sites_lst) > 0:
        for idx in original_binding_sites_lst:
            if isinstance(idx, int) and 1 <= idx <= sequence_len:  # Ensure index is valid
                binary_list[idx - 1] = 1

    return binary_list

def map_proteins_with_binding_sites(df_proteins, binding_files):
    """
    Map proteins with their respective ligand types and binding site positions.

    :param df_proteins: DataFrame containing protein IDs and sequences.
    :param binding_files: Dictionary mapping ligand types to their respective binding files.
    :return: DataFrame with ligand types and binding site positions.
    """
    binding_data = []

    for ligand_type, file_path in binding_files.items():
        with open(file_path, "r") as f:
            for line in f:
                protein_id, sites = line.strip().split("\t")
                binding_sites = [int(site) for site in sites.split(",")]
                binding_data.append({
                    "prot_id": protein_id,
                    "ligand_type": ligand_type,
                    "binding_sites": binding_sites
                })

    # Create a DataFrame from the binding data
    df_binding = pd.DataFrame(binding_data)

    # Merge the binding data with the protein DataFrame
    df_merged = pd.merge(df_proteins, df_binding, on="prot_id", how="left")

    return df_merged


def create_multi_label_binding_sites(df):
    # Initialize the new columns
    ligand_types = ['metal', 'small', 'nuclear']
    for ligand in ligand_types:
        df[f'{ligand}_binding'] = None

    # Fill in the binding sites for each ligand type
    for index, row in df.iterrows():
        seq_length = row['sequence_length']

        # Initialize empty arrays for each ligand type
        metal_binding = [0] * seq_length
        small_binding = [0] * seq_length
        nuclear_binding = [0] * seq_length

        # Extract binding sites and corresponding ligand types
        binding_sites = row['binding_sites']
        ligand_type = row['ligand_type']

        # For each binding site, mark the corresponding positions
        if ligand_type == "metal":
            metal_binding = convert_to_binary_list(binding_sites, seq_length)
        if ligand_type == "small":
            small_binding = convert_to_binary_list(binding_sites, seq_length)
        if ligand_type == "nuclear":
            nuclear_binding = convert_to_binary_list(binding_sites, seq_length)

        # Assign the arrays to the new columns
        df.at[index, 'metal_binding'] = metal_binding
        df.at[index, 'small_binding'] = small_binding
        df.at[index, 'nuclear_binding'] = nuclear_binding

    return df


def group_binding_sites(df):
    """
    Group binding sites by protein ID and aggregate relevant columns.

    :param df: DataFrame containing binding site information.
    :return: Grouped DataFrame with aggregated binding site data.
    """
    grouped_df = df.groupby('prot_id').agg({
        'binding_sites': lambda x: sorted(list(set(sum(x, [])))),  # Combine lists and remove duplicates
        'ligand_type': list,  # Collect all ligand types into a list
        'sequence': 'first',  # Keep the first sequence
        'sequence_length': 'first',  # Keep the first sequence_length
        'metal_binding': lambda x: [1 if any(row[i] == 1 for row in x) else 0 for i in range(len(next(iter(x))))],
        'small_binding': lambda x: [1 if any(row[i] == 1 for row in x) else 0 for i in range(len(next(iter(x))))],
        'nuclear_binding': lambda x: [1 if any(row[i] == 1 for row in x) else 0 for i in range(len(next(iter(x))))],
    }).reset_index()
    return grouped_df

def add_any_ligand_binding_sites(df):
    """
    Add a column 'any_ligand_binding_sites' to indicate binding sites (1) or non-binding sites (0),
    regardless of ligand type.

    :param df: DataFrame containing binding site information.
    :return: DataFrame with the new column added.
    """
    df['any_ligand_binding_sites'] = df.apply(
        lambda row: [1 if any(binding[i] == 1 for binding in [row['metal_binding'], row['small_binding'], row['nuclear_binding']]) else 0
                     for i in range(row['sequence_length'])],
        axis=1
    )
    return df


if __name__ == "__main__":
    test_315_df = load_graphddips_fasta_dataframe("data/development_set/Test_315.fa")
    test_315_df.to_csv("data/development_set/Test_315.csv", index=False, encoding="utf-8-sig")
    print("Data preparation completed successfully!")
    
    
