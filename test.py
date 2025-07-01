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

def strimed_labels_string_to_list(s):
    for char in s:
        if char.isnumeric() or char == ",":
            pos = s.index(char)
            break
    protein_str = s[:pos].strip()
    labels_str = s[pos:].strip()
    labels_list = [int(label.strip()) for label in labels_str if label.strip().isdigit()]
    return protein_str, labels_list

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

    protein_ids, proteins_lst, labels_lst = [], [], []

    for record in records:
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


    

file_path = "data/development_set/Train_335.fa"

train_initial_df = load_fasta_dataframe(file_path)
print(train_initial_df.head())

# records = [
#     (record.id, str(record.seq)) for record in SeqIO.parse(file_path, "fasta")
# ]

# sample_record = records[0]
# protein_str, labels_list = strimed_labels_string_to_list(sample_record[1])
# print(f"Protein ID: {sample_record[0]}")
# print(f"Protein Sequence: {protein_str}")
# print(f"Labels List: {labels_list}")