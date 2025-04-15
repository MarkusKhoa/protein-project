import requests
import pandas as pd
import numpy as np
import mdtraj as md
import torch

from loguru import logger
from Bio import PDB
from Bio.PDB.DSSP import DSSP
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

MAX_SASA = {
    'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0, 'CYS': 167.0,
    'GLU': 223.0, 'GLN': 225.0, 'GLY': 104.0, 'HIS': 224.0, 'ILE': 197.0,
    'LEU': 201.0, 'LYS': 236.0, 'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0,
    'SER': 155.0, 'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0
}

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
    """
    Extract amino acid types from a protein structure.
    """
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
    """
    Extract secondary structure features from a .PDB file using DSSP for Alphafold format.
    
    Args:
        structure (Bio.PDB.Structure): The protein structure object extracted from DSSP module.
        pdb_file (str): Path to the .PDB file.
    
    Returns:
        List: List of raw secondary structure codes (H, G, I, E, B, T, S, C).
            
    """
    model = structure[0]
    dssp = DSSP(model, pdb_file, dssp='/usr/bin/dssp')

    second_struct = []
    for residue in dssp:
        ss = residue[2]
        second_struct.append(ss)
    
    return second_struct

def get_secondary_structure_mdtraj(pdb_file):
    """
    Extract secondary structure features from a .PDB file using mdtraj.
    
    Args:
        pdb_file (str): Path to the .PDB file.
    
    Returns:
        dict: A dictionary with:
            - 'raw': List of raw secondary structure codes.
            - 'one_hot': Tensor of one-hot encoded secondary structure.
    """
    # Load the .PDB file with mdtraj
    traj = md.load(pdb_file)
    
    # Compute secondary structure
    ss = md.compute_dssp(traj)[0]  # Returns codes like 'H', 'E', 'C', 'NA'
    
    # One-hot encode
    ss_onehot = []
    for code in ss:
        ss_onehot.append([
            1 if code == 'H' else 0,
            1 if code == 'E' else 0,
            1 if code == 'T' else 0,
            1 if code == 'C' or code == 'NA' else 0
        ])
    
    ss_onehot = torch.tensor(ss_onehot, dtype=torch.float32)

    return {
        "raw": ss.tolist(),
        "one_hot": ss_onehot
    }

def calculate_residue_distances(coordinates):
    """
    Calculate pairwise distances between residues in a protein structure.
    Args: Coordinates (list): List of residue atom's coordinates.
    Returns:
        np.ndarray: 2D array of pairwise distances.
    """
    num_residues = len(coordinates)
    distances = np.zeros((num_residues, num_residues))

    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            distances[i, j] = distances[j, i] = dist
    
    return distances

def compute_rsa(structure_file, chain_id=None):
    """
    Compute RSA for each residue in a protein structure.

    Args:
        structure_file (str): Path to the protein structure file (e.g., PDB file).
        chain_id (str, optional): Chain ID to analyze (if None, uses the first chain).

    Returns:
        list: List of (residue_name, residue_id, rsa) tuples.
    """
    # Step 2: Load the protein structure
    traj = md.load(structure_file)
    
    # If a specific chain is requested, filter the topology
    if chain_id:
        chain = next(c for c in traj.topology.chains if c.chain_id == chain_id)
        traj = traj.atom_slice([atom.index for atom in traj.topology.atoms if atom.residue.chain == chain])

    # Step 3: Compute SASA per residue
    sasa = md.shrake_rupley(traj, mode='residue')[0]  # Shape: [n_residues]

    # Step 4: Compute RSA for each residue
    rsa_values = []
    for i, residue in enumerate(traj.topology.residues):
        # Skip non-amino acid residues (e.g., water, ligands)
        if not residue.is_protein:
            continue

        # Get the residue name (e.g., 'ALA', 'ARG')
        res_name = residue.name
        res_id = residue.resSeq

        # Get the computed SASA for this residue
        residue_sasa = sasa[i]

        # Get the maximum SASA for this residue type
        max_sasa = MAX_SASA.get(res_name, 0.0)
        if max_sasa == 0.0:
            print(f"Warning: No max SASA value for residue {res_name}. Skipping.")
            continue

        # Compute RSA
        rsa = residue_sasa / max_sasa if max_sasa > 0 else 0.0
        rsa = min(rsa, 1.0)  # Cap RSA at 1.0 (in case of numerical errors)

        rsa_values.append((res_name, res_id, rsa))

    return rsa_values

def get_dihedral_angles(pdb_file, sequence_length):
    traj = md.load(pdb_file)
    # Compute phi and psi angles
    phi_indices, phi_angles = md.compute_phi(traj)
    psi_indices, psi_angles = md.compute_psi(traj)

    # Convert to degrees and create tensors
    phi_angles = torch.tensor(np.degrees(phi_angles[0]), dtype=torch.float32).unsqueeze(1)  # Shape: [num_residues-1, 1]
    psi_angles = torch.tensor(np.degrees(psi_angles[0]), dtype=torch.float32).unsqueeze(1)  # Shape: [num_residues-1, 1]

    # Pad with zeros to match sequence length
    while phi_angles.shape[0] < sequence_length:
        phi_angles = torch.cat([phi_angles, torch.zeros(1, 1, dtype=torch.float32)], dim=0)
    while psi_angles.shape[0] < sequence_length:
        psi_angles = torch.cat([psi_angles, torch.zeros(1, 1, dtype=torch.float32)], dim=0)

    return phi_angles, psi_angles

def get_b_factors(structure, sequence_length):
    b_factors = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_atom = residue["CA"]
                    b_factor = ca_atom.get_bfactor()
                    b_factors.append(b_factor)

    # Pad with zeros if b_factors length is less than sequence length
    while len(b_factors) < sequence_length:
        b_factors.append(0.0)

    return torch.tensor(b_factors, dtype=torch.float32).unsqueeze(1)


train_df = pd.read_csv("data/development_set/full_grouped_train_binding_sites_df.csv")
test_df = pd.read_csv("data/development_set/full_grouped_test_binding_sites_df.csv")

idx = 100

sample_uniprot_id = train_df.iloc[idx]['prot_id']
sequence_protein = train_df.iloc[idx]['sequence']
sequence_length = len(sequence_protein)

print(f"Prot ID: {sample_uniprot_id}")

# uniprot_id = "P12345"  # Example UniProt ID
pdb_file = f"data/pdb_files/{sample_uniprot_id}_alphafold.pdb"
protein_structure = get_structure(sample_uniprot_id, pdb_file)

coordinates = extract_coordinates(protein_structure)
amino_acids = get_amino_acid_types(protein_structure)
secondary_structure = get_secondary_structure(protein_structure, pdb_file)
ss_onehot = get_secondary_structure_mdtraj(pdb_file)["one_hot"]
phi_angles, psi_angles = get_dihedral_angles(pdb_file, sequence_length)
b_factors = get_b_factors(protein_structure, sequence_length)
num_residues = len(coordinates)
print(f"Number of residues: {num_residues}")
min_length = min(num_residues, ss_onehot.shape[0], phi_angles.shape[0], psi_angles.shape[0], b_factors.shape[0])
residue_distances = calculate_residue_distances(coordinates)
rsa_values_list = compute_rsa(pdb_file)
# print(f"RSA values list: {rsa_values_list}")
residue_ids = [residue[1] for residue in rsa_values_list]
# print(f"Residue IDs: {residue_ids}")

rsa_dict = {res_id: rsa for _, res_id, rsa in rsa_values_list}
rsa_tensor = torch.zeros((min_length, 1), dtype=torch.float)
for i, res_id in enumerate(residue_ids[:min_length]):
    rsa_tensor[i, 0] = float(rsa_dict.get(res_id, 0.0))

print(f"rsa_tensor shape: {rsa_tensor.shape}")
print(f"RSA values: {rsa_tensor}")
# print(f"Amino acids: {amino_acids}")
# print(f"Secondary structure: {secondary_structure}")
# print(f"Sequence length: {len(sequence_protein)}")
# print(f"Secondary structures length: {len(secondary_structure)}")
# print(f"Residue distances: {residue_distances}")

