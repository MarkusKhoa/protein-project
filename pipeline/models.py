"""GNN model backbones for binding site prediction."""
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

try:
    from kan import KAN
except ImportError:
    KAN = None


def _make_kan_classifier(in_dim: int) -> nn.Module:
    """Create KAN-based classifier or fallback to Linear."""
    if KAN is not None:
        return KAN([in_dim, 8, 2], grid=2, k=2)
    return nn.Sequential(
        nn.Linear(in_dim, 32),
        nn.LeakyReLU(0.15),
        nn.Linear(32, 2),
    )


class BindingSiteGCN(nn.Module):
    """GCN backbone for binding site prediction."""

    def __init__(self, node_dim: int, edge_dim: int = 2, hidden_dim: int = 512):
        super().__init__()
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 256)
        self.conv3 = GCNConv(256, 128)
        self.edge_lin = nn.Linear(edge_dim, hidden_dim)
        self.pre_fc = nn.Linear(128, 16)
        self.fc = _make_kan_classifier(16)
        self.activation_func = nn.LeakyReLU(negative_slope=0.15)
        self.dropout = nn.Dropout(0.15)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = self.activation_func(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.activation_func(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.activation_func(x)
        return self.fc(x)


class BindingSiteGraphSAGE(nn.Module):
    """GraphSAGE backbone for binding site prediction."""

    def __init__(self, node_dim: int, edge_dim: int = 2, hidden_dim: int = 512):
        super().__init__()
        self.conv1 = SAGEConv(node_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, 256)
        self.conv3 = SAGEConv(256, 64)
        self.edge_lin = nn.Linear(edge_dim, hidden_dim)
        self.pre_fc = nn.Linear(64, 32)
        self.fc = _make_kan_classifier(32)
        self.activation_func = nn.LeakyReLU(negative_slope=0.15)
        self.dropout = nn.Dropout(0.15)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_attr = self.edge_lin(edge_attr)
        x = self.conv1(x, edge_index)
        x = self.activation_func(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.activation_func(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.activation_func(x)
        x = self.pre_fc(x)
        return self.fc(x)


class BindingSiteGraphSAGEWithBias(nn.Module):
    """GraphSAGE with binding-prone amino acid bias."""

    def __init__(self, node_dim: int, edge_dim: int = 2, hidden_dim: int = 512):
        super().__init__()
        self.conv1 = SAGEConv(node_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, 256)
        self.conv3 = SAGEConv(256, 64)
        self.pre_fc = nn.Linear(64, 32)
        self.fc = _make_kan_classifier(32)
        self.binding_prone_bias = nn.Parameter(torch.tensor(0.2))
        self.activation_func = nn.LeakyReLU(0.15)
        self.dropout = nn.Dropout(0.15)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        is_binding_prone = data.is_binding_prone.float().squeeze()
        if is_binding_prone.dim() == 0:
            is_binding_prone = is_binding_prone.unsqueeze(0).expand(x.shape[0])
        if is_binding_prone.shape[0] != x.shape[0]:
            raise RuntimeError(f"Shape mismatch: is_binding_prone ({is_binding_prone.shape}) vs x ({x.shape})")

        x = self.conv1(x, edge_index)
        x = self.activation_func(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.activation_func(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.activation_func(x)
        x = self.pre_fc(x)
        logits = self.fc(x)
        logits[:, 1] = logits[:, 1] + self.binding_prone_bias * is_binding_prone
        return logits


class BindingSiteGAT(nn.Module):
    """GAT backbone for binding site prediction."""

    def __init__(self, node_dim: int = 1287, hidden_dim: int = 512, heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(node_dim, hidden_dim, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, 256, heads=heads, concat=True)
        self.conv3 = GATConv(256 * heads, 128, heads=1, concat=False)
        self.pre_fc = nn.Linear(128, 16)
        self.fc = _make_kan_classifier(16)
        self.activation_func = nn.LeakyReLU(negative_slope=0.15)
        self.dropout = nn.Dropout(0.15)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.activation_func(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.activation_func(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.activation_func(x)
        return self.fc(x)


def build_model(
    backbone: str,
    node_dim: int,
    edge_dim: int = 2,
    hidden_dim: int = 512,
    gat_heads: int = 4,
    use_binding_bias: bool = True,
) -> nn.Module:
    """Build model by backbone name."""
    backbone = backbone.lower()
    if backbone == "gcn":
        return BindingSiteGCN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
    if backbone == "graphsage":
        if use_binding_bias:
            return BindingSiteGraphSAGEWithBias(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
        return BindingSiteGraphSAGE(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim)
    if backbone == "gat":
        return BindingSiteGAT(node_dim=node_dim, hidden_dim=hidden_dim, heads=gat_heads)
    raise ValueError(f"Unknown backbone: {backbone}. Use gcn, graphsage, or gat.")
