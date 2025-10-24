"""
utils/autoencoder.py
Non-linear dimensionality-reduction utilities
============================================

• SimpleAutoencoder  – tiny 2-layer linear AE
• fit_autoencoder    – train on a matrix (train-fold only) → params dict
• transform_features – use saved params to obtain 500-D codes
• scale_and_reduce_blocks_ae – drop-in replacement for scale_and_reduce_blocks()

Author: 2025-06-15  (updated 2025-07-09)
Python: 3.7  (CUDA if available)
"""

from typing import List, Tuple, Dict
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler, StandardScaler


# --------------------------------------------------------------------------- #
#                        constants & lightweight helpers                      #
# --------------------------------------------------------------------------- #

CANONICAL_ORDER = ("binding", "cat", "ec", "global")


def group_blocks(names: List[str],
                 blocks: List[np.ndarray]) -> Dict[str, List[np.ndarray]]:
    """
    Bucket feature blocks by semantic tag.

    Returns
    -------
    Dict[str, List[np.ndarray]]
        Keys are any of 'binding', 'cat', 'ec', 'global'
        that actually occur in *names*; values preserve original order.
    """
    groups: Dict[str, List[np.ndarray]] = {}
    for nm, blk in zip(names, blocks):
        lower = nm.lower()
        for tag in CANONICAL_ORDER:
            if tag in lower:
                groups.setdefault(tag, []).append(blk)
                break
    return groups


def _scale_blocks(train_blocks: List[np.ndarray],
                  test_blocks:  List[np.ndarray],
                  do_scale:     bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust-scale each block individually, concatenate them,
    and finally Standard-scale the whole group.
    """
    if not train_blocks:          # group absent
        raise ValueError("Bug: _scale_blocks called with empty group.")

    if not do_scale:
        return (np.concatenate(train_blocks, axis=1),
                np.concatenate(test_blocks,  axis=1))

    tr_scaled, te_scaled = [], []
    for b_tr, b_te in zip(train_blocks, test_blocks):
        rob = RobustScaler().fit(b_tr)
        tr_scaled.append(rob.transform(b_tr))
        te_scaled.append(rob.transform(b_te))

    tr_cat = np.concatenate(tr_scaled, axis=1)
    te_cat = np.concatenate(te_scaled, axis=1)

    std = StandardScaler().fit(tr_cat)
    return std.transform(tr_cat), std.transform(te_cat)


# --------------------------------------------------------------------------- #
#                           model architecture                                #
# --------------------------------------------------------------------------- #

class SimpleAutoencoder(nn.Module):
    """Linear encoder to *latent_dim*, ReLU, linear decoder."""
    def __init__(self, input_dim: int, latent_dim: int = 500):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim, bias=True),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(x)


# --------------------------------------------------------------------------- #
#                       core training / inference utils                       #
# --------------------------------------------------------------------------- #

def fit_autoencoder(
    X_train: np.ndarray,
    latent_dim: int = 500,
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    """Train an AE on X_train and return a *serialisable* param-dict."""
    X = torch.tensor(X_train, dtype=torch.float32)
    dataset = TensorDataset(X)
    loader  = DataLoader(dataset, shuffle=True, batch_size=batch_size, drop_last=False)

    model = SimpleAutoencoder(X.shape[1], latent_dim).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse   = nn.MSELoss()

    model.train()
    for _ in range(n_epochs):
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            opt.zero_grad()
            mse(model(batch), batch).backward()
            opt.step()

    return {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "input_dim":  X.shape[1],
        "latent_dim": latent_dim,
    }


def _build_model_from_params(params: Dict, device: str):
    mdl = SimpleAutoencoder(params["input_dim"], params["latent_dim"])
    mdl.load_state_dict(params["state_dict"])
    return mdl.to(device).eval()


def transform_features(
    params: Dict,
    X_full: np.ndarray,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 4096,
) -> np.ndarray:
    """Encode **any** feature matrix using a pre-trained AE."""
    mdl = _build_model_from_params(params, device)
    ds  = DataLoader(
        TensorDataset(torch.tensor(X_full, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False, drop_last=False
    )
    latents = []
    with torch.no_grad():
        for (batch,) in ds:
            z = mdl.encode(batch.to(device, non_blocking=True))
            latents.append(z.cpu().numpy())
    return np.concatenate(latents, axis=0)


# --------------------------------------------------------------------------- #
#             drop-in replacement for scale_and_reduce_blocks()               #
# --------------------------------------------------------------------------- #

def scale_and_reduce_blocks_ae(
    blocks_train: List[np.ndarray],
    blocks_test:  List[np.ndarray],
    block_names:  List[str],
    latent_dim:   int   = 500,
    n_epochs:     int   = 100,
    batch_size:   int   = 256,
    lr:           float = 1e-4,
    scale:        bool  = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Auto-encoder analogue of scale_and_reduce_blocks().

    * A separate AE is trained **per semantic group** (binding, cat, ec, global)
      that actually exists in `block_names`.
    * Each AE is fitted on **train-fold data only**.
    * The returned matrices are the concatenation of group-wise latent
      codes in the canonical order

          binding → cat → ec → global
    """
    # 1️⃣ bucket blocks by tag
    tr_groups = group_blocks(block_names, blocks_train)
    te_groups = group_blocks(block_names, blocks_test)

    X_tr_parts, X_te_parts = [], []

    for tag in CANONICAL_ORDER:
        if tag not in tr_groups:
            continue          # this dataset has no blocks of that type

        # 2️⃣ scale with robust+standard pipeline (optional)
        X_tr_grp, X_te_grp = _scale_blocks(tr_groups[tag],
                                           te_groups[tag],
                                           do_scale=scale)

        # 3️⃣ train AE on the group (train-fold only)
        params = fit_autoencoder(X_tr_grp,
                                 latent_dim=latent_dim,
                                 n_epochs=n_epochs,
                                 batch_size=batch_size,
                                 lr=lr)

        # 4️⃣ obtain latent codes for both splits
        X_tr_z = transform_features(params, X_tr_grp,batch_size=batch_size)
        X_te_z = transform_features(params, X_te_grp,batch_size=batch_size)

        X_tr_parts.append(X_tr_z)
        X_te_parts.append(X_te_z)

    return np.concatenate(X_tr_parts, axis=1), np.concatenate(X_te_parts, axis=1)
