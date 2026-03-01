"""Training loop with hard-negative mining and checkpoint selection."""
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.optim as optim

from pipeline.evaluate import evaluate
from pipeline.losses import ImprovedPositionAwareLoss, add_binding_features


def train(
    model: torch.nn.Module,
    train_loader,
    eval_loader,
    test_loader,
    device: torch.device,
    config,
    criterion: Optional[torch.nn.Module] = None,
    is_adding_binding_prone: bool = True,
) -> tuple[dict, Optional[dict]]:
    """
    Run training loop. Returns (final_metrics, best_state_dict).
    """
    if criterion is None:
        criterion = ImprovedPositionAwareLoss(
            pos_weight=config.training.pos_weight,
            position_weight=config.training.position_weight,
        )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    num_epochs = config.training.num_epochs
    hard_negative_threshold = config.training.hard_negative_threshold
    val_threshold = config.training.val_threshold

    best_val_f1 = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            if is_adding_binding_prone:
                batch = add_binding_features(batch, device)

            optimizer.zero_grad()
            out = model(batch)

            probs = torch.softmax(out, dim=1)[:, 1]
            labels = batch.y

            hard_negative_mask = (labels == 0) & (probs > hard_negative_threshold)
            true_positive_mask = labels == 1
            indices_for_loss = torch.where(true_positive_mask | hard_negative_mask)[0]

            if len(indices_for_loss) > 0:
                loss = criterion(out[indices_for_loss], labels[indices_for_loss], batch)
            else:
                loss = criterion(out, labels, batch)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        val_metrics = evaluate(
            model,
            eval_loader,
            device,
            threshold=val_threshold,
            criterion=criterion,
            is_adding_binding_prone=is_adding_binding_prone,
        )

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f} | MCC: {val_metrics['mcc']:.4f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print("  -> Best F1, saving checkpoint.")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        threshold=config.training.test_threshold,
        criterion=criterion,
        is_adding_binding_prone=is_adding_binding_prone,
    )
    return test_metrics, best_model_state


def save_checkpoint(
    state_dict: dict,
    save_dir: Path,
    prefix: str = "best",
) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{prefix}_model.pth"
    torch.save(state_dict, path)
    return path
