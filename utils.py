import torch
import mdtraj as md
import esm
import time
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

start = time.time()
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)

model.esm = model.esm.half()
torch.backends.cuda.matmul.allow_tf32 = True

test_df = pd.read_csv("data/development_set/full_grouped_test_binding_sites_df.csv")

test_protein = test_df.iloc[0]['sequence']
tokenized_input = tokenizer([test_protein], return_tensors="pt", add_special_tokens=False)['input_ids']

with torch.no_grad():
    output = model(tokenized_input)

end = time.time()

print(f"Processing time: {round(end - start, 2)} seconds")
print(f"Model output: {output}")