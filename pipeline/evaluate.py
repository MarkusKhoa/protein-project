"""Evaluation metrics and inference."""
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from pipeline.losses import add_binding_features


def evaluate(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    threshold: float = 0.5,
    criterion: Optional[torch.nn.Module] = None,
    is_adding_binding_prone: bool = False,
) -> dict:
    """
    Evaluate model on data_loader. Returns dict of metrics.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            if is_adding_binding_prone:
                batch = add_binding_features(batch, device)

            out = model(batch)
            if criterion is not None:
                total_val_loss += criterion(out, batch.y, batch).item()

            probs = torch.softmax(out, dim=1)[:, 1]
            preds = (probs >= threshold).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_val_loss = total_val_loss / max(len(data_loader), 1)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    unique_labels = np.unique(all_labels)
    if len(unique_labels) < 2:
        return {
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "auc": np.nan,
            "mcc": np.nan,
            "auprc": np.nan,
            "val_loss": avg_val_loss,
        }

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "val_loss": avg_val_loss,
    }
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    metrics["precision"] = p
    metrics["recall"] = r
    metrics["f1"] = f1
    metrics["auc"] = roc_auc_score(all_labels, all_probs)
    metrics["mcc"] = matthews_corrcoef(all_labels, all_preds)
    metrics["auprc"] = average_precision_score(all_labels, all_probs)

    return metrics
