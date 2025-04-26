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
    # Initialize a dictionary to store binding site data
    binding_data = []

    # Iterate over each ligand type and its corresponding file
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
    # Example usage
    fasta_file_path = "data/independent_set/indep_set.fasta"
    test_binding_sites_file_path = "data/independent_set/indep_set.txt"

    # Load data
    df_proteins = load_fasta_dataframe(fasta_file_path)
    # df_binding_sites = load_binding_sites_dataframe(test_binding_sites_file_path)

    binding_files = {
        "metal": "data/independent_set/binding_residues_metal.txt",
        "nuclear": "data/independent_set/binding_residues_nuclear.txt",
        "small": "data/independent_set/binding_residues_small.txt"
    }

    # Map proteins with binding sites
    initial_labeled_proteins_df = map_proteins_with_binding_sites(df_proteins, binding_files)
    multi_labeled_proteins_df = create_multi_label_binding_sites(initial_labeled_proteins_df)
    grouped_labeled_proteins_df = group_binding_sites(multi_labeled_proteins_df)
    grouped_labeled_proteins_df = add_any_ligand_binding_sites(grouped_labeled_proteins_df)

    grouped_labeled_proteins_df.to_csv("data/independent_set/grouped_test_46_new_binding_sites.csv", index=False)
    
    # df_result.to_csv("data/independent_set/indep_set_binding_sites.csv", index=False)
    # display(df_result)

    # binding_sites_df = pd.read_csv("data/development_set/all_binding_sites_complete.csv")
    # binding_sites_df['binding_sites'] = binding_sites_df['binding_sites'].apply(ast.literal_eval)

    # testing_binding_sites_df = binding_sites_df.loc[(binding_sites_df['prot_id'].isin(test_ids))].reset_index()

    # # Create multi-label binding sites

    # full_test_binding_sites_df = create_multi_label_binding_sites(testing_binding_sites_df)

    # # grouped_train_binding_sites_df = group_binding_sites(full_train_binding_sites_df)
    # grouped_test_binding_sites_df = group_binding_sites(full_test_binding_sites_df)
    # grouped_test_binding_sites_df.to_csv("data/independent_set/grouped_test_46_new_binding_sites.csv", index=False)
    
