import requests
import pandas as pd
import numpy as np

from loguru import logger
from Bio import PDB
from Bio.PDB.DSSP import DSSP
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

def get_structure(prot_id, pdb_file):
    parser = PDB.PDBParser()
    structure = parser.get_structure(prot_id, pdb_file)
    return structure

def extract_coordinates(structure):
    # Extract Cα coordinates (central carbon atom)
    coordinates = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:  # Get Cα atom
                    ca_atom = residue["CA"]
                    coord = ca_atom.get_coord()  # Returns numpy array [x, y, z]
                    coordinates.append(coord)
    
    return coordinates

def get_amino_acid_types(structure):
    amino_acids = []
    for model in structure:
        for chain in model:
            for residue in chain:
                # Get residue name (3-letter code)
                if not is_aa(residue):
                    continue
                res_name = residue.get_resname()
                # Convert to 1-letter code if needed
                one_letter = seq1(res_name)
                amino_acids.append(one_letter)
    return amino_acids

def get_secondary_structure(structure, pdb_file):
    model = structure[0]
    dssp = DSSP(model, pdb_file, dssp='/usr/bin/dssp')

    second_struct = []
    for residue in dssp:
        ss = residue[2]
        second_struct.append(ss)
    
    return second_struct

def calculate_residue_distances(coordinates):
    num_residues = len(coordinates)
    distances = np.zeros((num_residues, num_residues))

    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            distances[i, j] = distances[j, i] = dist
    
    return distances


train_df = pd.read_csv("data/development_set/full_grouped_train_binding_sites_df.csv")
test_df = pd.read_csv("data/development_set/full_grouped_test_binding_sites_df.csv")

idx = 10

sample_uniprot_id = train_df.iloc[idx]['prot_id']
sequence_protein = train_df.iloc[idx]['sequence']

print(f"Prot ID: {sample_uniprot_id}")

# uniprot_id = "P12345"  # Example UniProt ID
pdb_file = f"data/pdb_files/{sample_uniprot_id}_alphafold.pdb"
protein_structure = get_structure(sample_uniprot_id, pdb_file)

coordinates = extract_coordinates(protein_structure)
amino_acids = get_amino_acid_types(protein_structure)
secondary_structure = get_secondary_structure(protein_structure, pdb_file)
residue_distances = calculate_residue_distances(coordinates)

print(f"Amino acids: {amino_acids}")
print(f"Secondary structure: {secondary_structure}")
print(f"Sequence length: {len(sequence_protein)}")
print(f"Secondary structures length: {len(secondary_structure)}")
print(f"Residue distances: {residue_distances}")


# pdb_file = download_alphafold_structure(uniprot_id)