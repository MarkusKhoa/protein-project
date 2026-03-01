from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from loguru import logger
from tqdm import tqdm
from Bio import SeqIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import multiprocessing
import pandas as pd
import os

options = Options()
options.add_argument("--headless")  # Run in headless mode
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def get_sequence_from_online_db(prot_id, driver):
    logger.info(f"Processing {prot_id} ID")
    info_prot_url = f"https://www.uniprot.org/uniprotkb/{prot_id}/entry#sequences"
    initial_url = info_prot_url
    driver.get(info_prot_url)
    driver.implicitly_wait(1)
    WebDriverWait(driver, 3).until(
                lambda d: d.execute_script('return document.readyState') == 'complete'
            )
    current_url = driver.current_url
    if current_url != initial_url:
        logger.error(f"This entry sequence: {info_prot_url} is no longer annotated in UniProtKB")
        return None
    try:
        button = WebDriverWait(driver, 3).until(
        EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Show sequence')]")))
        button.click()
    except:
        logger.warning(f"No need to click the button 'Show sequence' ")

    try:
        # Use explicit wait for sequence chunks
        sequence_chunks = WebDriverWait(driver, 3).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "sequence__chunk"))
        )
        final_sequence = "".join(chunk.text for chunk in sequence_chunks)
        return final_sequence
    except Exception as e:
        logger.error(f"Failed to find sequence chunks: {str(e)}")
        return None

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

def process_batch(batch_df, batch_num):
    """Process a single batch of proteins"""
    logger.info(f"Processing batch {batch_num}, size: {len(batch_df)}")
    
    try:
        # Create a new Chrome driver for this batch
        batch_driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        
        # Get sequence information from UniProt database online
        batch_df['sequence'] = [
            get_sequence_from_online_db(prot_id, batch_driver) if pd.isna(seq) or seq == "" else seq
            for prot_id, seq in tqdm(zip(batch_df['prot_id'], batch_df['sequence']), 
                                    total=len(batch_df), desc=f"Batch {batch_num} Sequences")
        ]

        batch_df['binding_sites'] = batch_df['binding_sites'].apply(lambda x: list(x))
        batch_df['sequence_length'] = batch_df['sequence'].apply(lambda x: len(x) if type(x) == str else 0)

        batch_df['binary_binding_sites'] = [
            convert_to_binary_list(row['binding_sites'], row['sequence_length']) 
            for _, row in batch_df.iterrows()
        ]
        
        # Save batch results
        batch_filename = f"data/development_set/missing_val_binding_sites_batch_{batch_num}.csv"
        batch_df.to_csv(batch_filename, index = False, encoding = "utf-8-sig")
        logger.info(f"Saved batch {batch_num}")
        
        # Clean up
        batch_driver.quit()
        
        return batch_num, batch_df
        
    except Exception as e:
        logger.error(f"Error processing batch {batch_num}: {str(e)}")
        return batch_num, None


def get_sequence_wrapper(prot_id):
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    try:
        seq = get_sequence_from_online_db(prot_id, driver)
    finally:
        driver.quit()
    return prot_id, seq

def main_pipeline():
    logger.info("Starting pipeline...")
    folder = "ligysis_downloads"
    filenames = os.listdir(folder)

    # Extract protein IDs
    protein_ids = [filename.split('_')[0] for filename in filenames if os.path.isfile(os.path.join(folder, filename))]

    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(get_sequence_wrapper, prot_id) for prot_id in protein_ids]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching sequences"):
            try:
                pid, seq = future.result()
                results[pid] = seq
                print(f"{pid}: {seq[:30] if seq else 'No sequence found'}...")
            except Exception as exc:
                print(f"A protein ID generated an exception: {exc}")
    # Optionally, save results to a file
    results_df = pd.DataFrame(list(results.items()), columns=['prot_id', 'sequence'])
    results_df.to_csv('data/development_set/ligysis_protein_sequences.csv', index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    main_pipeline()
    # Uncomment to run the binding sites extraction pipeline
    # extract_binding_sites_pipeline()
