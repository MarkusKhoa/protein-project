import os
import requests
import pandas as pd

from pathlib import Path
from loguru import logger
from tqdm import tqdm

class AlphaFoldDownloader:
    def __init__(self, output_path="data/pdb_files"):
        """Initialize AlphaFold structure downloader.
        
        Args:
            output_path (str): Directory to save PDB files
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def download_structure(self, prot_id):
        """Download single protein structure from AlphaFold.
        
        Args:
            prot_id (str): Protein ID to download
            
        Returns:
            str or None: Path to downloaded file if successful, None otherwise
        """
        base_url = f"https://alphafold.ebi.ac.uk/files/AF-{prot_id}-F1-model_v4.pdb"
        
        try:
            response = requests.get(base_url)
            response.raise_for_status()
            
            filename = self.output_path / f"{prot_id}_alphafold.pdb"
            with open(filename, "wb") as f:
                f.write(response.content)
                
            logger.info(f"Successfully downloaded structure for {prot_id}")
            return str(filename)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download structure for {prot_id}: {str(e)}")
            return None
            
    def download_multiple(self, prot_ids):
        """Download multiple protein structures.
        
        Args:
            prot_ids (list): List of protein IDs to download
            
        Returns:
            dict: Mapping of protein IDs to their downloaded file paths
        """
        results = {}

        start_idx = 1037
        remaining_ids = prot_ids[start_idx:]
        total_remaining = len(remaining_ids)

        with tqdm(total = total_remaining, desc=f"Downloading from index {start_idx}/{len(prot_ids)}") as pbar:
            for idx, prot_id in enumerate(remaining_ids, start=start_idx):
                filepath = self.download_structure(prot_id)
                results[prot_id] = filepath
                pbar.set_description(f"Processing {idx}/{len(prot_ids)}")
                pbar.update(1)
            
        return results


train_df = pd.read_csv("data/development_set/official_de_dup_training_sites_df.csv")
test_df = pd.read_csv("data/development_set/official_de_dup_testing_sites_df.csv")
needed_protein_ids = train_df.iloc[:]['prot_id'].tolist() + test_df.iloc[:]['prot_id'].tolist()

alphafold_downloader = AlphaFoldDownloader()
results = alphafold_downloader.download_multiple(needed_protein_ids)
