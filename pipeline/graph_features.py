"""Graph construction and structure feature extraction."""
from typing import Optional

import numpy as np
import pandas as pd
import torch
from Bio import PDB
from Bio.PDB import Selection
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1
from torch_geometric.data import Data
from tqdm import tqdm

from pipeline.io import resolve_pdb_path

MAX_SASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLU": 223.0, "GLN": 225.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}

HYDROPHOBICITY_SCALE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

POLARITY_MAP = {
    "A": 0, "C": 0, "D": 1, "E": 1, "F": 0,
    "G": 0, "H": 1, "I": 0, "K": 1, "L": 0,
    "M": 0, "N": 1, "P": 0, "Q": 1, "R": 1,
    "S": 1, "T": 1, "V": 0, "W": 0, "Y": 1,
}


def get_structure(prot_id: str, pdb_file: str):
    parser = PDB.PDBParser()
    return parser.get_structure(prot_id, pdb_file)


def extract_coordinates(structure):
    coordinates = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    coordinates.append(residue["CA"].get_coord())
    return coordinates


def get_amino_acid_types(structure):
    amino_acids = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if not is_aa(residue):
                    continue
                amino_acids.append(seq1(residue.get_resname()))
    return amino_acids


def get_secondary_structure_mdtraj(pdb_file: str, sequence_length: int):
    import mdtraj as md

    traj = md.load(pdb_file)
    ss = md.compute_dssp(traj)[0]

    ss_onehot = []
    for code in ss:
        ss_onehot.append([
            1 if code == "H" else 0,
            1 if code == "E" else 0,
            1 if code == "T" else 0,
            1 if code == "C" or code == "NA" else 0,
        ])
    ss_onehot = torch.tensor(ss_onehot, dtype=torch.float32)

    if ss_onehot.shape[0] < sequence_length:
        padding = torch.zeros((sequence_length - ss_onehot.shape[0], ss_onehot.shape[1]), dtype=torch.float32)
        ss_onehot = torch.cat([ss_onehot, padding], dim=0)

    return {"raw": ss.tolist(), "one_hot": ss_onehot}


def calculate_residue_distances(coordinates):
    num_residues = len(coordinates)
    distances = np.zeros((num_residues, num_residues))
    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            distances[i, j] = distances[j, i] = dist
    return distances


def get_dihedral_angles(pdb_file: str, sequence_length: int):
    import mdtraj as md

    traj = md.load(pdb_file)
    _, phi_angles = md.compute_phi(traj)
    _, psi_angles = md.compute_psi(traj)

    phi_angles = torch.tensor(np.degrees(phi_angles[0]), dtype=torch.float32).unsqueeze(1)
    psi_angles = torch.tensor(np.degrees(psi_angles[0]), dtype=torch.float32).unsqueeze(1)

    while phi_angles.shape[0] < sequence_length:
        phi_angles = torch.cat([phi_angles, torch.zeros(1, 1, dtype=torch.float32)], dim=0)
    while psi_angles.shape[0] < sequence_length:
        psi_angles = torch.cat([psi_angles, torch.zeros(1, 1, dtype=torch.float32)], dim=0)
    return phi_angles, psi_angles


def get_hydrophobicity(structure):
    amino_acids = get_amino_acid_types(structure)
    hydro = [HYDROPHOBICITY_SCALE.get(aa, 0.0) for aa in amino_acids]
    return torch.tensor(hydro, dtype=torch.float32).unsqueeze(1)


def get_polarity(structure):
    amino_acids = get_amino_acid_types(structure)
    polar = [POLARITY_MAP.get(aa, 0) for aa in amino_acids]
    return torch.tensor(polar, dtype=torch.float32).unsqueeze(1)


def compute_rsa(structure_file: str, chain_id: Optional[str] = None) -> torch.Tensor:
    import mdtraj as md

    traj = md.load(structure_file)
    if chain_id:
        chain = next(c for c in traj.topology.chains if c.chain_id == chain_id)
        traj = traj.atom_slice([atom.index for atom in traj.topology.atoms if atom.residue.chain == chain])

    sasa = md.shrake_rupley(traj, mode="residue")[0]
    rsa_values = []
    for i, residue in enumerate(traj.topology.residues):
        if not residue.is_protein:
            continue
        res_name = residue.name
        residue_sasa = sasa[i]
        max_sasa = MAX_SASA.get(res_name, 0.0)
        if max_sasa == 0.0:
            continue
        rsa = min(residue_sasa / max_sasa, 1.0)
        rsa_values.append(rsa)
    return torch.tensor(rsa_values, dtype=torch.float32).unsqueeze(1)


def fuse_features(
    esm2_embeddings: torch.Tensor,
    ss_onehot: torch.Tensor,
    phi_angles: torch.Tensor,
    psi_angles: torch.Tensor,
    rsa_values: torch.Tensor,
    hydrophobicity: torch.Tensor,
    polarity: torch.Tensor,
) -> torch.Tensor:
    num_residues = esm2_embeddings.shape[0]
    min_length = min(
        num_residues,
        ss_onehot.shape[0],
        phi_angles.shape[0],
        psi_angles.shape[0],
        rsa_values.shape[0],
        hydrophobicity.shape[0],
        polarity.shape[0],
    )

    esm2_embeddings = esm2_embeddings[:min_length]
    ss_onehot = ss_onehot[:min_length]
    phi_angles = phi_angles[:min_length]
    psi_angles = psi_angles[:min_length]
    rsa_values = rsa_values[:min_length]
    hydrophobicity = hydrophobicity[:min_length]
    polarity = polarity[:min_length]

    return torch.cat([
        esm2_embeddings,
        ss_onehot,
        phi_angles,
        psi_angles,
        rsa_values,
        hydrophobicity,
        polarity,
    ], dim=1)


def create_edge_features(distances: np.ndarray, threshold: float = 8.0):
    num_residues = distances.shape[0]
    contact_map = (distances < threshold) & (distances > 0)
    edge_index = torch.nonzero(torch.tensor(contact_map, dtype=torch.bool), as_tuple=False).t()

    src, dst = edge_index[0], edge_index[1]
    dists = torch.tensor(distances[src.numpy(), dst.numpy()], dtype=torch.float32)
    seq_seps = torch.abs(src - dst).float()
    dists = dists / threshold
    seq_seps = seq_seps / num_residues
    edge_attr = torch.stack([dists, seq_seps], dim=1)
    return edge_index, edge_attr


def create_graph_data(node_features, edge_index, edge_attr, labels) -> Data:
    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=labels,
    )


def get_graph_data(
    df: pd.DataFrame,
    embeddings: list,
    device: torch.device,
    pdb_dir: str,
    label_column: str = "labels",
) -> list:
    graph_data_list = []
    for idx in tqdm(range(len(df)), desc="Building graphs"):
        prot_id = df.iloc[idx]["id"]
        sequence_len = len(df.iloc[idx]["sequence"])
        labels = df.iloc[idx][label_column]
        embedding = torch.tensor(embeddings[idx]["embeddings"])

        try:
            structure_file = str(resolve_pdb_path(prot_id, pdb_dir))
            structure = get_structure(prot_id, structure_file)
            coordinates = extract_coordinates(structure)
            distances = calculate_residue_distances(coordinates)
            rsa_values = compute_rsa(structure_file)
            ss_one_hot = get_secondary_structure_mdtraj(structure_file, sequence_len)["one_hot"]
            phi_angles, psi_angles = get_dihedral_angles(structure_file, sequence_len)
            hydrophobicity = get_hydrophobicity(structure).to(device)
            polarity = get_polarity(structure).to(device)
        except Exception as e:
            print(f"Skipping {prot_id}: {e}")
            continue

        min_len = min(
            embedding.shape[0],
            ss_one_hot.shape[0],
            phi_angles.shape[0],
            psi_angles.shape[0],
            rsa_values.shape[0],
            hydrophobicity.shape[0],
            polarity.shape[0],
        )
        if rsa_values.shape[0] < min_len:
            rsa_pad = torch.zeros(min_len - rsa_values.shape[0], 1, dtype=torch.float32)
            rsa_values = torch.cat([rsa_values, rsa_pad], dim=0)

        node_features = fuse_features(
            embedding[:min_len].to(device),
            ss_one_hot[:min_len].to(device),
            phi_angles[:min_len].to(device),
            psi_angles[:min_len].to(device),
            rsa_values[:min_len].to(device),
            hydrophobicity[:min_len].to(device),
            polarity[:min_len].to(device),
        )
        edge_index, edge_attr = create_edge_features(distances)
        labels_t = torch.tensor(labels[:min_len], dtype=torch.long).to(device)
        graph_data = create_graph_data(node_features, edge_index, edge_attr, labels_t)
        graph_data_list.append(graph_data)

    return graph_data_list
