import os
import math
import copy
import hashlib
from collections import defaultdict, deque
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score


# GROUPING: connected components (leakage prevention)
def compute_components(df: pd.DataFrame) -> np.ndarray:
    adj = defaultdict(set)
    s_list = df["Sire"].astype(str).values
    d_list = df["Dam"].astype(str).values

    for s, d in zip(s_list, d_list):
        adj[f"S:{s}"].add(f"D:{d}")
        adj[f"D:{d}"].add(f"S:{s}")

    visited = set()
    comp_map = {}
    cid = 0
    for node in list(adj.keys()):
        if node in visited:
            continue
        q = deque([node])
        visited.add(node)
        comp_nodes = set()
        while q:
            u = q.popleft()
            comp_nodes.add(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        for n in comp_nodes:
            comp_map[n] = cid
        cid += 1

    comp_id = np.empty(len(df), dtype=int)
    for i, (s, d) in enumerate(zip(s_list, d_list)):
        comp_id[i] = comp_map.get(f"S:{s}", comp_map.get(f"D:{d}", -1))
    return comp_id


def signature_from_crossids(cids) -> str:
    arr = np.array(sorted(map(str, cids)), dtype=str)
    return hashlib.sha256(("|".join(arr)).encode("utf-8")).hexdigest()[:16]


# DATASET
class FertilityDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feat_table: pd.DataFrame, id_col: str = "CrossID"):
        idx_s = df["Sire"].astype(str).values
        idx_d = df["Dam"].astype(str).values

        Xs = feat_table.reindex(idx_s).to_numpy(np.float32, copy=False)
        Xd = feat_table.reindex(idx_d).to_numpy(np.float32, copy=False)

        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        Xd = np.nan_to_num(Xd, nan=0.0, posinf=0.0, neginf=0.0)

        self.sire = torch.from_numpy(Xs)
        self.dam  = torch.from_numpy(Xd)
        self.y    = torch.from_numpy(df["BinPheno"].values.astype(np.float32))
        self.ids  = df[id_col].astype(str).values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.sire[i], self.dam[i], self.y[i], self.ids[i]


# CV folds (cached)
def build_cv_folds(
    df: pd.DataFrame,
    *,
    splits_dir: str,
    n_folds: int,
    seed: int,
    tag: str,
) -> Tuple[Dict, str]:
    """
    Returns (parts, cache_path) where parts["cv_fold_id"] maps CrossID->fold_id.

    NOTE: df must already have the final BinPheno convention applied (e.g. flipped).
    """
    os.makedirs(splits_dir, exist_ok=True)

    df = df.copy()
    df["CrossID"] = df["CrossID"].astype(str)

    sig = signature_from_crossids(df["CrossID"].values)
    cache_path = os.path.join(splits_dir, f"cv{n_folds}_{tag}_{sig}.pkl")

    if os.path.exists(cache_path):
        parts = torch.load(cache_path)
        if "cv_fold_id" not in parts:
            raise RuntimeError(f"Bad cache file (missing cv_fold_id): {cache_path}")
        return parts, cache_path

    df["comp_id"] = compute_components(df)
    y = df["BinPheno"].astype(int).to_numpy()
    g = df["comp_id"].to_numpy()
    cids = df["CrossID"].to_numpy()

    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_id = {}
    for k, (_, va_idx) in enumerate(cv.split(np.zeros(len(df)), y, g)):
        for cid in cids[va_idx]:
            fold_id[str(cid)] = int(k)

    if set(fold_id.keys()) != set(map(str, cids)):
        missing = set(map(str, cids)) - set(fold_id.keys())
        extra = set(fold_id.keys()) - set(map(str, cids))
        raise RuntimeError(f"Fold map mismatch. missing={len(missing)} extra={len(extra)}")

    parts = {"cv_fold_id": fold_id, "meta": {"n_folds": n_folds, "seed": seed, "tag": tag, "sig": sig}}
    torch.save(parts, cache_path)
    return parts, cache_path


def add_fold_id(df: pd.DataFrame, parts: dict) -> pd.DataFrame:
    df = df.copy()
    df["CrossID"] = df["CrossID"].astype(str)
    fold_map = parts["cv_fold_id"]
    if set(df["CrossID"].values) != set(fold_map.keys()):
        raise RuntimeError("CrossID mismatch vs cached fold map (won't run to keep reproducibility honest).")
    df["fold_id"] = df["CrossID"].map(fold_map).astype(int)
    return df


# ------------------------------
# Probit-product loss & metrics
def weighted_bce_from_logp(log_p, y, pos_weight_value: float):
    y = y.float()

    # sanitize log_p BEFORE exp / log1p
    log_p = torch.nan_to_num(log_p, nan=-30.0, neginf=-30.0, posinf=0.0)

    p = log_p.exp().clamp(1e-12, 1.0 - 1e-12)
    log_one_minus_p = torch.log1p(-p)

    per = -(y * log_p + (1 - y) * log_one_minus_p)

    pos_w = torch.as_tensor(pos_weight_value, dtype=torch.float32, device=y.device)
    w = torch.where(y > 0.5, pos_w, torch.ones_like(y))
    return (per * w).sum() / w.sum().clamp_min(1.0)


def pos_weight_from_df(df: pd.DataFrame) -> float:
    n_pos = int((df["BinPheno"] == 1).sum())
    n_neg = int((df["BinPheno"] == 0).sum())
    return float(n_neg) / max(1.0, float(n_pos))


@torch.no_grad()
def eval_auc_pr_probit_product(model, loader, device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    probs, labels = [], []

    for sire_x, dam_x, y, _ in loader:
        sire_x = sire_x.to(device)
        dam_x  = dam_x.to(device)

        out = model(sire_x, dam_x)
        if isinstance(out, (tuple, list)) and len(out) == 3:
            _, _, log_p = out
        else:
            raise RuntimeError("Expected model to return (a, b, log_p)")

        p = torch.exp(log_p)
        p = torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
        p = p.clamp(1e-7, 1 - 1e-7)

        probs.extend(p.detach().cpu().numpy().ravel())
        labels.extend(y.detach().cpu().numpy().ravel())

    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probs, dtype=np.float64)
    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    pr  = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    return auc, pr

@torch.no_grad()
def eval_auc_pr_compat(model, loader, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    probs, labels = [], []

    for sire_x, dam_x, y, _ in loader:
        sire_x = sire_x.to(device)
        dam_x = dam_x.to(device)

        hs, hd, log_p = model(sire_x, dam_x)

        p = torch.exp(log_p)
        p = torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
        p = p.clamp(1e-7, 1.0 - 1e-7)

        probs.extend(p.detach().cpu().numpy().ravel())
        labels.extend(y.detach().cpu().numpy().ravel())

    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probs, dtype=np.float64)

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    pr = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan

    return auc, pr

@torch.no_grad()
def collect_tower_latents_probit_product(model, loader, device: Optional[str] = None) -> pd.DataFrame:
    """
    Collects per-sample tower liabilities (a,b)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    rows = []

    for batch in loader:
        
        sire_x, dam_x, y, cross_ids = batch

        sire_x = sire_x.to(device)
        dam_x  = dam_x.to(device)

        a, b, log_p = model(sire_x, dam_x)

        p = torch.exp(log_p)
        p = torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0).clamp(1e-7, 1.0 - 1e-7)

        a_np = a.detach().cpu().numpy().ravel().astype(float)
        b_np = b.detach().cpu().numpy().ravel().astype(float)
        p_np = p.detach().cpu().numpy().ravel().astype(float)
        ids  = np.asarray(cross_ids).astype(str)

        for cid, aa, bb, pp in zip(ids, a_np, b_np, p_np):
            rows.append({"CrossID": cid, "sire_tower": aa, "dam_tower": bb, "p_hat": pp})

    return pd.DataFrame(rows)


# Training loops (w early stopping)
def train_es_probit_product(
    model,
    train_loader,
    val_loader,
    *,
    pos_weight_value: float,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    max_grad_norm: float,
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_score = -float("inf")
    best_state = None
    wait = 0

    for _epoch in range(1, int(max_epochs) + 1):
        model.train()
        for sire_x, dam_x, y, _ in train_loader:
            sire_x = sire_x.to(device)
            dam_x  = dam_x.to(device)
            y      = y.to(device)

            _, _, log_p = model(sire_x, dam_x)

            loss = weighted_bce_from_logp(log_p, y, pos_weight_value)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm:
                nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            opt.step()

        auc, pr = eval_auc_pr_probit_product(model, val_loader, device=device)
        score = auc if (auc == auc) else pr

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_es_lasso_probit_product(
    model,
    train_loader,
    val_loader,
    *,
    pos_weight_value: float,
    lr: float,
    l1_lambda: float,
    max_epochs: int,
    patience: int,
    max_grad_norm: float,
    device: Optional[str] = None,
):
    """
    plus L1 penalty
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=0.0)

    best_score = -float("inf")
    best_state = None
    wait = 0

    for _epoch in range(1, int(max_epochs) + 1):
        model.train()
        for sire_x, dam_x, y, _ in train_loader:
            sire_x = sire_x.to(device)
            dam_x  = dam_x.to(device)
            y      = y.to(device)

            _, _, log_p = model(sire_x, dam_x)

            loss = weighted_bce_from_logp(log_p, y, pos_weight_value)

            # L1 penalty (only weights)
            l1 = model.sire.weight.abs().sum() + model.dam.weight.abs().sum()
            loss = loss + float(l1_lambda) * l1

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm:
                nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            opt.step()

            if hasattr(model, "proximal_step"):
                model.proximal_step(lr=float(lr), l1_lambda=float(l1_lambda), include_bias=False)

        auc, pr = eval_auc_pr_probit_product(model, val_loader, device=device)
        score = auc if (auc == auc) else pr

        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def train_es_compat(
    model,
    train_loader,
    val_loader,
    *,
    pos_weight_value: float,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    max_grad_norm: float,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr= float(lr),
        weight_decay= float(weight_decay),
    )

    best_score = -float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, int(max_epochs) + 1):
        model.train()

        for sire_x, dam_x, y, _ in train_loader:
            sire_x = sire_x.to(device)
            dam_x = dam_x.to(device)
            y = y.to(device)

            hs, hd, log_p = model(sire_x, dam_x)

            loss = weighted_bce_from_logp(
                log_p,
                y,
                pos_weight_value,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if max_grad_norm:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(max_grad_norm),
                )

            opt.step()

        auc, pr = eval_auc_pr_compat(
            model,
            val_loader,
            device=device,
        )

        score = auc if auc == auc else pr

        if score > best_score:
            best_score = score
            best_state = {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# Dataloaders helper
def make_loaders(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    feat_table: pd.DataFrame,
    *,
    batch_train: int,
    batch_eval: int,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    val_with_ids: bool = False,
):
    if num_workers is None:
        num_workers = 3 if torch.cuda.is_available() else 0
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    tr_loader = DataLoader(
        FertilityDataset(df_tr, feat_table),
        batch_size=int(batch_train),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=(int(num_workers) > 0),
    )

    val_ds = FertilityDataset(df_va, feat_table)
    va_loader = DataLoader(
        val_ds,
        batch_size=int(batch_eval),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=(int(num_workers) > 0),
    )
    return tr_loader, va_loader
