"""Central configuration for the pipeline."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import yaml
except ImportError:
    yaml = None  # optional


@dataclass
class PathConfig:
    """Paths for data and artifacts."""

    train_csv: str = "data/train_335_old.csv"
    test_csv: str = "data/development_set/test_60.csv"
    pdb_dir: str = "data/esmFold_pdb_files"
    save_dir: str = "artifacts"
    embeddings_cache_dir: Optional[str] = None


@dataclass
class EmbeddingConfig:
    """ESM-2 embedding configuration."""

    model_name: str = "facebook/esm2_t33_650M_UR50D"
    max_sequence_length: int = 1000
    batch_size: int = 32
    embedding_mode: str = "multi_layer"
    num_layers: int = 2


@dataclass
class ModelConfig:
    """GNN model configuration."""

    backbone: str = "graphsage"  # graphsage, gcn, gat
    hidden_dim: int = 512
    edge_dim: int = 2
    gat_heads: int = 4
    use_binding_bias: bool = True  # For GraphSAGE with binding-prone bias


@dataclass
class TrainingConfig:
    """Training configuration."""

    num_epochs: int = 70
    batch_size: int = 64
    learning_rate: float = 0.0001
    weight_decay: float = 0.015
    pos_weight: float = 7.0
    position_weight: float = 1.15
    hard_negative_threshold: float = 0.6
    val_split: float = 0.02
    val_threshold: float = 0.62
    test_threshold: float = 0.635


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""

    paths: PathConfig = field(default_factory=PathConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
    excluded_protein_ids: List[str] = field(default_factory=lambda: ["Q9NZV6"])
    label_column: str = "labels"
    id_column: str = "id"

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load config from YAML file."""
        if yaml is None:
            raise ImportError("PyYAML is required for config. Install with: pip install pyyaml")
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        paths = PathConfig(**(raw.get("paths") or {}))
        embedding = EmbeddingConfig(**(raw.get("embedding") or {}))
        model = ModelConfig(**(raw.get("model") or {}))
        training = TrainingConfig(**(raw.get("training") or {}))

        return cls(
            paths=paths,
            embedding=embedding,
            model=model,
            training=training,
            seed=raw.get("seed", 42),
            excluded_protein_ids=raw.get("excluded_protein_ids", ["Q9NZV6"]),
            label_column=raw.get("label_column", "labels"),
            id_column=raw.get("id_column", "id"),
        )

    def to_dict(self) -> dict:
        """Export config as dict for logging."""
        return {
            "paths": {
                "train_csv": self.paths.train_csv,
                "test_csv": self.paths.test_csv,
                "pdb_dir": self.paths.pdb_dir,
                "save_dir": self.paths.save_dir,
            },
            "embedding": {
                "model_name": self.embedding.model_name,
                "max_sequence_length": self.embedding.max_sequence_length,
                "batch_size": self.embedding.batch_size,
                "embedding_mode": self.embedding.embedding_mode,
                "num_layers": self.embedding.num_layers,
            },
            "model": {
                "backbone": self.model.backbone,
                "hidden_dim": self.model.hidden_dim,
                "edge_dim": self.model.edge_dim,
                "gat_heads": self.model.gat_heads,
                "use_binding_bias": self.model.use_binding_bias,
            },
            "training": {
                "num_epochs": self.training.num_epochs,
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "pos_weight": self.training.pos_weight,
                "position_weight": self.training.position_weight,
                "hard_negative_threshold": self.training.hard_negative_threshold,
                "val_split": self.training.val_split,
                "val_threshold": self.training.val_threshold,
                "test_threshold": self.training.test_threshold,
            },
            "seed": self.seed,
            "excluded_protein_ids": self.excluded_protein_ids,
            "label_column": self.label_column,
            "id_column": self.id_column,
        }
