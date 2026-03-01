#!/usr/bin/env python3
"""
End-to-end pipeline for protein binding site prediction.
Supports GraphSAGE, GCN, GAT and other GNN backbones.
"""
import argparse
import gc
import pickle
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from pipeline.config import PipelineConfig
from pipeline.embeddings import get_embeddings_list, tokenize_sequences
from pipeline.evaluate import evaluate
from pipeline.graph_features import get_graph_data
from pipeline.io import load_train_test
from pipeline.losses import ImprovedPositionAwareLoss, add_binding_features
from pipeline.models import build_model
from pipeline.repro import get_device, log_run_metadata, set_seed
from pipeline.train import save_checkpoint, train
from transformers import EsmTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Run protein binding site prediction pipeline")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config")
    p.add_argument("--model", type=str, choices=["graphsage", "gcn", "gat"], help="GNN backbone")
    p.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--save-dir", type=str, default=None, help="Output directory for artifacts")
    p.add_argument("--smoke", action="store_true", help="Smoke test with minimal data")
    return p.parse_args()


def main():
    args = parse_args()

    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig()

    if args.model:
        config.model.backbone = args.model
    if args.seed is not None:
        config.seed = args.seed
    if args.save_dir:
        config.paths.save_dir = args.save_dir

    set_seed(config.seed)
    device = get_device() if args.device is None else torch.device(args.device)

    # Load data
    train_df, test_df = load_train_test(
        config.paths.train_csv,
        config.paths.test_csv,
        label_column=config.label_column,
        excluded_protein_ids=config.excluded_protein_ids,
        id_column=config.id_column,
    )

    if args.smoke:
        train_df = train_df.head(4)
        test_df = test_df.head(2)

    train_seq = train_df["sequence"].tolist()
    test_seq = test_df["sequence"].tolist()

    # Tokenize
    tokenizer = EsmTokenizer.from_pretrained(config.embedding.model_name)
    train_tokenized = tokenize_sequences(
        train_seq,
        tokenizer,
        max_length=config.embedding.max_sequence_length,
    )
    test_tokenized = tokenize_sequences(
        test_seq,
        tokenizer,
        max_length=config.embedding.max_sequence_length,
    )

    # Embeddings (optional cache)
    emb_device = "cuda" if device.type == "cuda" else "cpu"
    cache_dir = config.paths.embeddings_cache_dir
    cache_key = f"embeddings_{config.seed}_{len(train_df)}_{len(test_df)}.pkl"
    if cache_dir:
        cache_path = Path(cache_dir) / cache_key
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            train_embeddings = cached["train"]
            test_embeddings = cached["test"]
            print(f"Loaded cached embeddings from {cache_path}")
        else:
            train_embeddings = None
    else:
        train_embeddings = None

    if train_embeddings is None:
        train_embeddings = get_embeddings_list(
            train_tokenized,
            batch_size=config.embedding.batch_size,
            model_name=config.embedding.model_name,
            device=emb_device,
            embedding_mode=config.embedding.embedding_mode,
            num_layers=config.embedding.num_layers,
            return_hidden_states=True,
            return_attentions=False,
        )
        test_embeddings = get_embeddings_list(
            test_tokenized,
            batch_size=config.embedding.batch_size,
            model_name=config.embedding.model_name,
            device=emb_device,
            embedding_mode=config.embedding.embedding_mode,
            num_layers=config.embedding.num_layers,
            return_hidden_states=True,
            return_attentions=False,
        )
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump({"train": train_embeddings, "test": test_embeddings}, f)
            print(f"Cached embeddings to {cache_path}")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Build graphs
    train_graphs = get_graph_data(
        train_df,
        train_embeddings,
        device,
        config.paths.pdb_dir,
        label_column=config.label_column,
    )
    test_graphs = get_graph_data(
        test_df,
        test_embeddings,
        device,
        config.paths.pdb_dir,
        label_column=config.label_column,
    )

    # Split train/val
    train_graphs, val_graphs = train_test_split(
        train_graphs,
        test_size=config.training.val_split,
        random_state=config.seed,
    )

    train_loader = DataLoader(train_graphs, batch_size=config.training.batch_size, shuffle=True)
    eval_loader = DataLoader(val_graphs, batch_size=config.training.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=config.training.batch_size, shuffle=False)

    # Build model: ESM + struct features (ss, phi, psi, rsa, hydro, polarity) + binding_propensity
    base_dim = train_embeddings[0]["embeddings"].shape[1]
    struct_dim = 4 + 1 + 1 + 1 + 1 + 1
    binding_extra = 1  # add_binding_features adds 1 column
    node_dim = base_dim + struct_dim + binding_extra

    use_binding_bias = config.model.use_binding_bias and config.model.backbone.lower() == "graphsage"
    # Always add binding features (adds 1 col to x); GraphSAGEWithBias also uses is_binding_prone
    add_binding_prone = True
    model = build_model(
        backbone=config.model.backbone,
        node_dim=node_dim,
        edge_dim=config.model.edge_dim,
        hidden_dim=config.model.hidden_dim,
        gat_heads=config.model.gat_heads,
        use_binding_bias=use_binding_bias,
    ).to(device)

    criterion = ImprovedPositionAwareLoss(
        pos_weight=config.training.pos_weight,
        position_weight=config.training.position_weight,
    )

    # Train
    metrics, best_state = train(
        model,
        train_loader,
        eval_loader,
        test_loader,
        device,
        config,
        criterion=criterion,
        is_adding_binding_prone=add_binding_prone,
    )

    # Save artifacts
    save_dir = Path(config.paths.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if best_state is not None:
        ckpt_path = save_checkpoint(best_state, save_dir, prefix=f"{config.model.backbone}_best")
        print(f"Saved checkpoint: {ckpt_path}")

    meta_path = log_run_metadata(save_dir, config.to_dict(), metrics)
    print(f"Run metadata: {meta_path}")
    print("Final test metrics:", metrics)


if __name__ == "__main__":
    main()
