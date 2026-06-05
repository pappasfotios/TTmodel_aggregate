import os
import glob
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit

from TTmodels import (
    BinaryFertilityModel,
    BinaryFertilityCompat,
    BinaryFertilityLassoTwoTower,
)

from TT_util import (
    compute_components,
    FertilityDataset,
    pos_weight_from_df,
    train_es_lasso_probit_product,
    eval_auc_pr_probit_product,
    train_es_probit_product,
    train_es_compat,
    eval_auc_pr_compat,
)

# Configurations
SEED = 39
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 3 if torch.cuda.is_available() else 0
PIN_MEMORY = torch.cuda.is_available()

BASE_1 = "HER_M0.4_F0.4"

BASE_DIR = "."
SCENARIOS_DIR = os.path.join(BASE_DIR, "scenarios")
PREFIX = "SimFeat_"

SPLITS_DIR = os.path.join(BASE_DIR, "splits_cache")
os.makedirs(SPLITS_DIR, exist_ok=True)

BATCH_TRAIN = 128
BATCH_EVAL = 512
MAX_GRAD_NORM = 5.0
MAX_EPOCHS = 100
PATIENCE = 7

HYPER_CSV = "/home/fotios/FertSim/optuna_best_all3_base_15k_probit_product.csv"


# Extra helper functions
def make_or_load_splits(crosses_df: pd.DataFrame, tag: str, seed: int = SEED):
    split_path = os.path.join(SPLITS_DIR, f"splits_{tag}_seed{seed}_flipY.npz")

    try:
        data = np.load(split_path)
        return {k: data[k].astype(int) for k in ("tr_idx", "va_idx", "te_idx")}
    except FileNotFoundError:
        pass

    tmp = crosses_df.copy()
    tmp["comp_id"] = compute_components(tmp)

    groups = tmp["comp_id"].to_numpy()
    y = tmp["BinPheno"].astype(int).to_numpy()

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


def make_loader(df_part, feat_table, batch_size, shuffle):
    return DataLoader(
        FertilityDataset(df_part, feat_table),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )


def discover_scenarios():
    files = sorted({
        os.path.basename(f)[len(PREFIX):]
        for f in glob.glob(os.path.join(SCENARIOS_DIR, f"{PREFIX}*"))
        if os.path.basename(f).startswith(PREFIX)
        and len(os.path.basename(f)) > len(PREFIX)
    })
    return [s.removesuffix(".csv") for s in files]



# Hyperparameters
hp_df = pd.read_csv(HYPER_CSV)
hp_df["model"] = hp_df["model"].astype(str).str.lower()

hp_mlp = hp_df[hp_df["model"] == "mlp"].iloc[0]
hp_lasso = hp_df[hp_df["model"] == "lasso"].iloc[0]


# Main
SCENARIOS = [s for s in discover_scenarios() if s.split("_R")[0] == BASE_1]

print(f"DEVICE: {DEVICE}", flush=True)
print(f"Found {len(SCENARIOS)} scenarios.", flush=True)

results_rows = []

for system in ["base", "compat"]: # 
    for tag in SCENARIOS:

        feat_path = os.path.join(SCENARIOS_DIR, f"SimFeat_{tag}.csv")

        if system == "base":
            phen_path = os.path.join(SCENARIOS_DIR, f"SimPhen_{tag}.csv")
            split_tag = tag + "_base_successY"
        else:
            phen_path = os.path.join(SCENARIOS_DIR, f"SimPhenCompat_{tag}.csv")
            split_tag = tag + "_compat_successY"

        if not os.path.exists(phen_path):
            print(f"Skipping missing: {phen_path}", flush=True)
            continue

        print(f"{system.upper()} SCENARIO: {tag}", flush=True)

        A_df = pd.read_csv(feat_path)
        A_df = A_df.rename(columns={A_df.columns[0]: "ID"})
        A_df["ID"] = A_df["ID"].astype(str)
        feat_table = A_df.set_index("ID").astype(np.float32)

        crosses_df = pd.read_csv(phen_path)

        if system == "base":
            required_cols = ["Sire", "Dam", "CrossID", "BinPheno"]
            missing = [c for c in required_cols if c not in crosses_df.columns]
            if missing:
                raise ValueError(f"{phen_path} missing required columns: {missing}")

            # original file convention: 1=failure, 0=success
            crosses_df["BinPheno"] = 1 - crosses_df["BinPheno"].astype(int)

        else:
            required_cols = [
                "Sire", "Dam", "CrossID",
                "BinPheno_Compat", "CompatFail", "LabelChanged",
            ]
            missing = [c for c in required_cols if c not in crosses_df.columns]
            if missing:
                raise ValueError(f"{phen_path} missing required columns: {missing}")

            # compat file convention: 1=failure, 0=success
            crosses_df["BinPheno"] = 1 - crosses_df["BinPheno_Compat"].astype(int)

        splits = make_or_load_splits(crosses_df, split_tag, seed=SEED)

        df_tr = crosses_df.iloc[splits["tr_idx"]].reset_index(drop=True).copy()
        df_va = crosses_df.iloc[splits["va_idx"]].reset_index(drop=True).copy()
        df_te = crosses_df.iloc[splits["te_idx"]].reset_index(drop=True).copy()

        pos_w = pos_weight_from_df(df_tr)

        tr_loader = make_loader(df_tr, feat_table, BATCH_TRAIN, True)
        va_loader = make_loader(df_va, feat_table, BATCH_EVAL, False)
        te_loader = make_loader(df_te, feat_table, BATCH_EVAL, False)

        compat_fail_rate = (
            float(crosses_df["CompatFail"].mean())
            if "CompatFail" in crosses_df.columns
            else np.nan)

        label_changed_rate = (
            float(crosses_df["LabelChanged"].mean())
            if "LabelChanged" in crosses_df.columns
            else np.nan)

        # Regular TT-MLP
        print(f"\n--- Training {system.upper()} REGULAR TT-MLP ---", flush=True)

        mlp = BinaryFertilityModel(
            input_dim=feat_table.shape[1],
            size=int(hp_mlp["size"]),
            dropout_rate=float(hp_mlp["dropout_rate"]),
        )

        mlp = train_es_probit_product(
            mlp,
            tr_loader,
            va_loader,
            pos_weight_value=pos_w,
            lr=float(hp_mlp["lr"]),
            weight_decay=float(hp_mlp["weight_decay"]),
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            max_grad_norm=MAX_GRAD_NORM,
            device=DEVICE,
        )

        mlp_auc, mlp_pr = eval_auc_pr_probit_product(
            mlp,
            te_loader,
            device=DEVICE,
        )

        print(
            f"[{system.upper()}_REGULAR_TT_MLP] TEST auc={mlp_auc:.3f} pr={mlp_pr:.3f}",
            flush=True,
        )

        results_rows.append({
            "scenario": tag,
            "system": system,
            "model": "regular_mlp",
            "test_auc": mlp_auc,
            "test_pr_auc": mlp_pr,
            "compat_fail_rate": compat_fail_rate,
            "label_changed_rate": label_changed_rate,
            "n_subtrain": len(df_tr),
            "n_val": len(df_va),
            "n_test": len(df_te),
        })

        # Concat MLP
        print(f"\n--- Training {system.upper()} CONCAT MLP ---", flush=True)

        concat_mlp = BinaryFertilityCompat(
            input_dim=feat_table.shape[1],
            size=int(hp_mlp["size"]),
            dropout_rate=float(hp_mlp["dropout_rate"]),
        )

        concat_mlp = train_es_compat(
            concat_mlp,
            tr_loader,
            va_loader,
            pos_weight_value=pos_w,
            lr=float(hp_mlp["lr"]),
            weight_decay=float(hp_mlp["weight_decay"]),
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            max_grad_norm=MAX_GRAD_NORM,
            device=DEVICE,
        )

        concat_auc, concat_pr = eval_auc_pr_compat(
            concat_mlp,
            te_loader,
            device=DEVICE,
        )

        results_rows.append({
            "scenario": tag,
            "system": system,
            "model": "concat_mlp",
            "test_auc": concat_auc,
            "test_pr_auc": concat_pr,
            "compat_fail_rate": compat_fail_rate,
            "label_changed_rate": label_changed_rate,
            "n_subtrain": len(df_tr),
            "n_val": len(df_va),
            "n_test": len(df_te),
        })

        # TT-LASSO
        print(f"\n--- Training {system.upper()} LASSO ---", flush=True)

        lasso = BinaryFertilityLassoTwoTower(
            input_dim=feat_table.shape[1],
        )

        lasso = train_es_lasso_probit_product(
            lasso,
            tr_loader,
            va_loader,
            pos_weight_value=pos_w,
            lr=float(hp_lasso["lr"]),
            l1_lambda=float(hp_lasso["l1_lambda"]),
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            max_grad_norm=MAX_GRAD_NORM,
            device=DEVICE,
        )

        lasso_auc, lasso_pr = eval_auc_pr_probit_product(
            lasso,
            te_loader,
            device=DEVICE,
        )

        results_rows.append({
            "scenario": tag,
            "system": system,
            "model": "lasso",
            "test_auc": lasso_auc,
            "test_pr_auc": lasso_pr,
            "compat_fail_rate": compat_fail_rate,
            "label_changed_rate": label_changed_rate,
            "n_subtrain": len(df_tr),
            "n_val": len(df_va),
            "n_test": len(df_te),
        })


# Save summary
res_df = pd.DataFrame(results_rows)

out_path = os.path.join(BASE_DIR, "compat_only_lasso_vs_mlp_summary.csv")
res_df.to_csv(out_path, index=False)