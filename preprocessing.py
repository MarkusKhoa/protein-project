from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from loguru import logger
from tqdm import tqdm
from Bio import SeqIO

import pandas as pd

options = Options()
options.add_argument("--headless")  # Run in headless mode
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def get_sequence_from_online_db(prot_id):
    url = f"https://www.uniprot.org/uniprotkb/{prot_id}/entry#sequences"
    driver.get(url)
    
    # Wait for the browser finish loading all elements
    driver.implicitly_wait(3)

    # Find all tags whose class name is `sequence__chunk`
    sequence_chunks = driver.find_elements(By.CLASS_NAME, "sequence__chunk")
    final_sequence = "".join(chunk.text for chunk in sequence_chunks)
    return final_sequence

def parse_fasta(fasta_file):
    seq_dict = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq_dict[record.id] = str(record.seq)
    return seq_dict

def parse_labels(label_file, ligand_type):
    binding_labels = []
    with open(label_file, "r") as f:
        for line in f:
            protein_id, indices = line.strip().split("\t")
            binding_labels.append((protein_id, set(map(int, indices.split(",")))))
    
    binding_labels_df = pd.DataFrame(binding_labels, columns=['prot_id', 'binding_sites'])
    binding_labels_df['ligand_type'] = ligand_type
    return binding_labels_df

def convert_to_binary_list(original_binding_sites_lst, sequence_len):
    """Convert a Binding-Active site string to a binary list based on the sequence length."""
    binary_list = [0] * sequence_len  # Initialize a list of zeros
    
    # Ensure original_binding_sites_lst is a list and not empty
    if isinstance(original_binding_sites_lst, list) and len(original_binding_sites_lst) > 0:
        for idx in original_binding_sites_lst:
            if isinstance(idx, int) and 1 <= idx <= sequence_len:  # Ensure index is valid
                binary_list[idx - 1] = 1

    return binary_list

def main_pipeline():
    fasta_file = "data/development_set/all.fasta"
    metal_label_file = "data/development_set/binding_residues_2.5_metal.txt"
    nuclear_label_file = "data/development_set/binding_residues_2.5_nuclear.txt"
    small_label_file = "data/development_set/binding_residues_2.5_small.txt"

    logger.info("Starting pipeline...")

    seq_dict = parse_fasta(fasta_file)
    metal_binding_sites_df = parse_labels(metal_label_file, 'metal')
    nuclear_binding_sites_df = parse_labels(nuclear_label_file, 'nuclear')
    small_binding_sites_df = parse_labels(small_label_file, 'small')
    all_binding_sites_df = pd.concat([metal_binding_sites_df, nuclear_binding_sites_df, small_binding_sites_df],
                                 ignore_index=True)
    
    logger.info(f"Total proteins to process: {len(all_binding_sites_df)}")
    
    # Map available sequence in dataset
    all_binding_sites_df['sequence'] = all_binding_sites_df['prot_id'].map(seq_dict)
    
    # Get sequence information from UniProt database online
    logger.info("Fetching missing sequences from UniProt...")

    all_binding_sites_df['sequence'] = [
        get_sequence_from_online_db(prot_id) if pd.isna(seq) or seq == "" else seq
        for prot_id, seq in tqdm(zip(all_binding_sites_df['prot_id'], all_binding_sites_df['sequence']), 
                                total=len(all_binding_sites_df), desc="Retrieving Sequences")
    ]

    all_binding_sites_df['binding_sites'] = all_binding_sites_df['binding_sites'].apply(lambda x: list(x))
    all_binding_sites_df['sequence_length'] = all_binding_sites_df['sequence'].apply(lambda x: len(x) if type(x) == str else 0)

    all_binding_sites_df['binary_binding_sites'] = [
        convert_to_binary_list(row['binding_sites'], row['sequence_length']) 
        for _, row in tqdm(all_binding_sites_df.iterrows(), total=len(all_binding_sites_df), desc="Processing Binary Sites")
    ]

    all_binding_sites_df.to_csv("data/development_set/all_binding_sites.csv", index = False, encoding="utf-8-sig")
    logger.info("Pipeline executed successfully!")
    

if __name__ == "__main__":
    main_pipeline()
