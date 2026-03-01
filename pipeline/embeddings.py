"""ESM-2 embeddings extraction."""
from typing import List

import torch
from tqdm import tqdm
from transformers import EsmModel, EsmTokenizer


def tokenize_sequences(
    sequences: List[str],
    tokenizer: EsmTokenizer,
    max_length: int = 1000,
) -> dict:
    """Tokenize protein sequences."""
    tokenized = tokenizer(
        sequences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        is_split_into_words=False,
    )
    return tokenized


def get_embeddings_list(
    tokenized_dataset: dict,
    batch_size: int,
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    device: str = "cuda",
    embedding_mode: str = "multi_layer",
    num_layers: int = 4,
    return_hidden_states: bool = True,
    return_attentions: bool = False,
) -> List[dict]:
    """
    Extract ESM-2 embeddings with multi-layer aggregation or attention-guided pooling.

    Args:
        tokenized_dataset: Dict with 'input_ids', 'attention_mask', and optionally 'sequence_id'.
        batch_size: Number of sequences per batch.
        model_name: Pretrained ESM-2 model name.
        device: Device to run model on ('cuda' or 'cpu').
        embedding_mode: 'multi_layer' for layer aggregation, 'attention_guided' for attention pooling.
        num_layers: Number of layers to aggregate (for multi_layer mode).
        return_hidden_states: Whether to return hidden states.
        return_attentions: Whether to return attention weights (required for attention_guided).

    Returns:
        List of dicts with sequence_id and per-residue embeddings.
    """
    if embedding_mode == "attention_guided" and not return_attentions:
        raise ValueError("Attention-guided pooling requires return_attentions=True")

    model = EsmModel.from_pretrained(model_name).to(device)
    model.eval()

    ids_list = tokenized_dataset["input_ids"].to(device)
    attention_mask_list = tokenized_dataset["attention_mask"].to(device)
    sequence_ids = tokenized_dataset.get("sequence_id", list(range(len(ids_list))))

    num_batches = (len(ids_list) + batch_size - 1) // batch_size
    embeddings_list = []

    for i in tqdm(range(num_batches), total=num_batches, desc="ESM-2 embeddings"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(ids_list))

        batch_ids = ids_list[start_idx:end_idx]
        batch_attention_mask = attention_mask_list[start_idx:end_idx]
        batch_seq_ids = sequence_ids[start_idx:end_idx]

        with torch.no_grad():
            outputs = model(
                input_ids=batch_ids,
                attention_mask=batch_attention_mask,
                output_hidden_states=return_hidden_states,
                output_attentions=(embedding_mode == "attention_guided"),
            )

            if embedding_mode == "multi_layer":
                hidden_states = outputs.hidden_states[-num_layers:]
                aggregated_embeddings = torch.stack(hidden_states, dim=0).mean(dim=0)

                for j in range(aggregated_embeddings.shape[0]):
                    mask = batch_attention_mask[j].bool()
                    seq_embeddings = aggregated_embeddings[j][mask][1:-1]
                    embeddings_list.append({
                        "sequence_id": batch_seq_ids[j],
                        "embeddings": seq_embeddings.cpu().numpy(),
                    })

            elif embedding_mode == "attention_guided":
                hidden_states = outputs.hidden_states[-1]
                attentions = outputs.attentions[-1]

                for j in range(hidden_states.shape[0]):
                    mask = batch_attention_mask[j].bool()
                    seq_hidden = hidden_states[j][mask]
                    seq_attention = attentions[j][:, mask][:, :, mask]
                    attention_weights = seq_attention.mean(dim=0)
                    attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-8)
                    weighted_embedding = torch.matmul(attention_weights, seq_hidden)
                    seq_embeddings = weighted_embedding[1:-1]
                    embeddings_list.append({
                        "sequence_id": batch_seq_ids[j],
                        "embeddings": seq_embeddings.cpu().numpy(),
                    })

            else:
                raise ValueError("embedding_mode must be 'multi_layer' or 'attention_guided'")

    return embeddings_list
