import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score

from TTmodels import BinaryFertilityModel, BinaryFertilityModelConv, BinaryFertilityLassoTwoTower

from TT_util import (
    compute_components,
    FertilityDataset,
    collect_tower_latents_probit_product,
    pos_weight_from_df,
    eval_auc_pr_probit_product,
    train_es_probit_product,
    train_es_lasso_probit_product,
)

# Configurations
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 3 if torch.cuda.is_available() else 0
PIN_MEMORY = torch.cuda.is_available()

BASE_DIR = "."
SCENARIOS_DIR = os.path.join(BASE_DIR, "scenarios")
PREFIX = "SimFeat_"

SPLITS_DIR = os.path.join(BASE_DIR, "splits_cache")
os.makedirs(SPLITS_DIR, exist_ok=True)

BATCH_TRAIN, BATCH_EVAL = 128, 512
MAX_GRAD_NORM = 5.0
MAX_EPOCHS = 100
PATIENCE = 7

# Extra helpers for data partitioning and evaluation
def make_or_load_splits(crosses_df: pd.DataFrame, tag: str, seed: int = SEED):
    split_path = os.path.join(SPLITS_DIR, f"splits_{tag}_seed{seed}_y1isfail.npz")

    if os.path.exists(split_path):
        data = np.load(split_path)
        return {k: data[k].astype(int) for k in ("tr_idx", "va_idx", "te_idx")}

    tmp = crosses_df.copy()
    tmp["comp_id"] = compute_components(tmp)

    groups = tmp["comp_id"].values
    y = tmp["BinPheno"].astype(int).values

    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, te_idx = next(outer.split(np.zeros(len(y)), y, groups))

    inner = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed + 1000)
    tr_rel, va_rel = next(inner.split(np.zeros(len(tr_idx)), y[tr_idx], groups[tr_idx]))

    splits = {
        "tr_idx": tr_idx[tr_rel],
        "va_idx": tr_idx[va_rel],
        "te_idx": te_idx,
    }

    np.savez(split_path, **splits)
    return splits


@torch.no_grad()
def eval_acc_only_probit_product(model, loader, device=DEVICE, threshold=0.5) -> float:
    model.eval()
    probs, labels = [], []
    for sire_x, dam_x, y, _ in loader:
        sire_x = sire_x.to(device)
        dam_x = dam_x.to(device)
        _, _, log_p = model(sire_x, dam_x)

        p = torch.exp(log_p)
        p = torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0).clamp(1e-7, 1.0 - 1e-7)

        probs.extend(p.detach().cpu().numpy().ravel())
        labels.extend(y.detach().cpu().numpy().ravel())

    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probs, dtype=np.float64)
    y_pred = (y_prob >= float(threshold)).astype(np.int64)
    return float(accuracy_score(y_true, y_pred))


def make_loader(df_part: pd.DataFrame, feat_table: pd.DataFrame, batch_size: int, shuffle: bool):
    return DataLoader(
        FertilityDataset(df_part, feat_table),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )


# Scenario discovery
FILES = sorted({
    os.path.basename(f)[len(PREFIX):]
    for f in glob.glob(os.path.join(SCENARIOS_DIR, f"{PREFIX}*"))
    if os.path.basename(f).startswith(PREFIX) and len(os.path.basename(f)) > len(PREFIX)
})
SCENARIOS = [s.removesuffix(".csv") for s in FILES]

# Load optimized hyperparameters
HYPER_CSV = "/home/fotios/FertSim/optuna_best_all3_base_15k_probit_product.csv"
hp_df = pd.read_csv(HYPER_CSV)
hp_df["model"] = hp_df["model"].astype(str).str.lower()

hp_mlp = hp_df[hp_df["model"] == "mlp"].iloc[0]
hp_conv = hp_df[hp_df["model"] == "conv"].iloc[0]
hp_lasso = hp_df[hp_df["model"] == "lasso"].iloc[0]


### Main loop ###
results_rows = []

for tag in SCENARIOS:

    print(f"SCENARIO: {tag}")

    feat_path = os.path.join(SCENARIOS_DIR, f"SimFeat_{tag}.csv")
    phen_path = os.path.join(SCENARIOS_DIR, f"SimPhen_{tag}.csv")

    # features
    A_df = pd.read_csv(feat_path)
    A_df = A_df.rename(columns={A_df.columns[0]: "ID"})
    A_df["ID"] = A_df["ID"].astype(str)
    feat_table = A_df.set_index("ID").astype(np.float32)

    # phenotypes
    crosses_df = pd.read_csv(phen_path)

    crosses_df["BinPheno"] = crosses_df["BinPheno"].astype(int)

    # splits
    splits = make_or_load_splits(crosses_df, tag, seed=SEED)
    df_tr = crosses_df.iloc[splits["tr_idx"]].reset_index(drop=True).copy()
    df_va = crosses_df.iloc[splits["va_idx"]].reset_index(drop=True).copy()
    df_te = crosses_df.iloc[splits["te_idx"]].reset_index(drop=True).copy()

    print(f"Prevalence (sub/val/test): "
          f"{1-df_tr['BinPheno'].mean():.3f} / {1-df_va['BinPheno'].mean():.3f} / {1-df_te['BinPheno'].mean():.3f}")

    pos_w = pos_weight_from_df(df_tr)

    tr_loader = make_loader(df_tr, feat_table, batch_size=BATCH_TRAIN, shuffle=True)
    va_loader = make_loader(df_va, feat_table, batch_size=BATCH_EVAL, shuffle=False)
    te_loader = make_loader(df_te, feat_table, batch_size=BATCH_EVAL, shuffle=False)

    # TT-MLP
    print("\n--- Training MLP ---")
    mlp = BinaryFertilityModel(
        input_dim=feat_table.shape[1],
        size=int(hp_mlp["size"]),
        dropout_rate=float(hp_mlp["dropout_rate"]),
    )
    mlp = train_es_probit_product(
        mlp, tr_loader, va_loader,
        pos_weight_value=pos_w,
        lr=float(hp_mlp["lr"]),
        weight_decay=float(hp_mlp["weight_decay"]),
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        max_grad_norm=MAX_GRAD_NORM,
        device=DEVICE,
    )

    te_auc, te_pr = eval_auc_pr_probit_product(mlp, te_loader, device=DEVICE)
    te_acc = eval_acc_only_probit_product(mlp, te_loader, device=DEVICE)
    print(f"[MLP] TEST acc={te_acc:.3f} auc={te_auc:.3f} pr={te_pr:.3f}")
    results_rows.append({
        "scenario": tag, "model": "mlp",
        "test_acc": te_acc, "test_auc": te_auc, "test_pr_auc": te_pr,
        "n_subtrain": len(df_tr), "n_val": len(df_va), "n_test": len(df_te),
    })

    tower_df = collect_tower_latents_probit_product(mlp, te_loader, device=DEVICE)
    truth_df = df_te[["CrossID", "SireBV", "DamBV"]].copy()
    merged = pd.merge(tower_df, truth_df, on="CrossID", how="left", validate="one_to_one")

    merged.insert(0, "scenario", tag)
    merged.insert(1, "model", "mlp")
    merged.to_csv(os.path.join(BASE_DIR, f"per_sample_tower_latents_{tag}_mlp.csv"), index=False)

    # TT-CNN
    # -------------------------
    print("\n--- Training CONV ---")
    conv = BinaryFertilityModelConv(
        input_channels=feat_table.shape[1],
        conv=int(hp_conv["conv"]),
        pool=int(hp_conv["pool"]),
        size=int(hp_conv["size"]),
        dropout_rate=float(hp_conv["dropout_rate"]),
    )
    conv = train_es_probit_product(
        conv, tr_loader, va_loader,
        pos_weight_value=pos_w,
        lr=float(hp_conv["lr"]),
        weight_decay=float(hp_conv["weight_decay"]),
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        max_grad_norm=MAX_GRAD_NORM,
        device=DEVICE,
    )

    te_auc, te_pr = eval_auc_pr_probit_product(conv, te_loader, device=DEVICE)
    te_acc = eval_acc_only_probit_product(conv, te_loader, device=DEVICE)
    print(f"[CONV] TEST acc={te_acc:.3f} auc={te_auc:.3f} pr={te_pr:.3f}")
    results_rows.append({
        "scenario": tag, "model": "conv",
        "test_acc": te_acc, "test_auc": te_auc, "test_pr_auc": te_pr,
        "n_subtrain": len(df_tr), "n_val": len(df_va), "n_test": len(df_te),
    })

    tower_df = collect_tower_latents_probit_product(conv, te_loader, device=DEVICE)
    truth_df = df_te[["CrossID", "SireBV", "DamBV"]].copy()
    merged = pd.merge(tower_df, truth_df, on="CrossID", how="left", validate="one_to_one")

    merged.insert(0, "scenario", tag)
    merged.insert(1, "model", "conv")
    merged.to_csv(os.path.join(BASE_DIR, f"per_sample_tower_latents_{tag}_conv.csv"), index=False)

    # TT-LASSO
    print("\n--- Training LASSO ---")
    lasso = BinaryFertilityLassoTwoTower(input_dim=feat_table.shape[1])
    lasso = train_es_lasso_probit_product(
        lasso, tr_loader, va_loader,
        pos_weight_value=pos_w,
        lr=float(hp_lasso["lr"]),
        l1_lambda=float(hp_lasso["l1_lambda"]),
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        max_grad_norm=MAX_GRAD_NORM,
        device=DEVICE,
    )

    te_auc, te_pr = eval_auc_pr_probit_product(lasso, te_loader, device=DEVICE)
    te_acc = eval_acc_only_probit_product(lasso, te_loader, device=DEVICE)
    print(f"[LASSO] TEST acc={te_acc:.3f} auc={te_auc:.3f} pr={te_pr:.3f}")
    results_rows.append({
        "scenario": tag, "model": "lasso",
        "test_acc": te_acc, "test_auc": te_auc, "test_pr_auc": te_pr,
        "n_subtrain": len(df_tr), "n_val": len(df_va), "n_test": len(df_te),
    })

    tower_df = collect_tower_latents_probit_product(lasso, te_loader, device=DEVICE)
    truth_df = df_te[["CrossID", "SireBV", "DamBV"]].copy()
    merged = pd.merge(tower_df, truth_df, on="CrossID", how="left", validate="one_to_one")

    merged.insert(0, "scenario", tag)
    merged.insert(1, "model", "lasso")
    merged.to_csv(os.path.join(BASE_DIR, f"per_sample_tower_latents_{tag}_lasso.csv"), index=False)


# Save summary
res_df = pd.DataFrame(results_rows)
out_path = os.path.join(BASE_DIR, "fixed_split_all_models_summary_probit_product.csv")
res_df.to_csv(out_path, index=False)