import torch
from features_extraction import get_amino_acid_types

hydrophobicity_scale = {
    'A': 1.8,  'C': 2.5,  'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5,  'K': -3.9, 'L': 3.8,
    'M': 1.9,  'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2,  'W': -0.9, 'Y': -1.3
}

def get_hydrophobicity(structure):
    hydro = []
    amino_acids = get_amino_acid_types(structure)
    for amino_acid in amino_acids:
        try:
            hydro.append(hydrophobicity_scale.get(amino_acid, 0.0))
        except Exception as e:
            hydro.append(0.0)
    return torch.tensor(hydro, dtype=torch.float32).unsqueeze(1)


polarity_map = {
    'A': 0, 'C': 0, 'D': 1, 'E': 1, 'F': 0,
    'G': 0, 'H': 1, 'I': 0, 'K': 1, 'L': 0,
    'M': 0, 'N': 1, 'P': 0, 'Q': 1, 'R': 1,
    'S': 1, 'T': 1, 'V': 0, 'W': 0, 'Y': 1
}

def get_polarity(structure):
    polar = []
    amino_acids = get_amino_acid_types(structure)
    for amino_acid in amino_acids:
        try:
            polar.append(polarity_map.get(amino_acid, 0))
        except Exception as e:
            polar.append(0)
    return torch.tensor(polar, dtype=torch.float32).unsqueeze(1)
