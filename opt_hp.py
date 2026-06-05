import os
import math
import copy
import hashlib
from collections import defaultdict, deque

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

import optuna

from TT_util import (
    add_fold_id,
    build_cv_folds,
    compute_components,
    FertilityDataset,
    collect_tower_latents_probit_product,
    pos_weight_from_df,
    eval_auc_pr_probit_product,
    train_es_probit_product,
    train_es_lasso_probit_product,
)

from TTmodels import (
    BinaryFertilityModel,
    BinaryFertilityModelConv,
    BinaryFertilityLassoTwoTower,
)

# CONFIGURATIONS
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 3 if torch.cuda.is_available() else 0
PIN_MEMORY = torch.cuda.is_available()

BASE_DIR = "../"
SCEN_DIR = os.path.join(BASE_DIR, "hyperparam_opt")

TAG = "base_15k"
FEAT_PATH = os.path.join(SCEN_DIR, f"SimFeat_{TAG}.csv")
PHEN_PATH = os.path.join(SCEN_DIR, f"SimPhen_{TAG}.csv")

N_FOLDS = 5
BATCH_TRAIN = 128
BATCH_EVAL = 512

MAX_EPOCHS = 100
PATIENCE = 7
MAX_GRAD_NORM = 5.0

SPLITS_DIR = os.path.join(BASE_DIR, "splits_cache_cv5")
os.makedirs(SPLITS_DIR, exist_ok=True)

# Optuna settings
N_TRIALS_MLP = 40
N_TRIALS_CONV = 60
N_TRIALS_LASSO = 20

SAMPLER = optuna.samplers.TPESampler(seed=SEED, multivariate=True, group=True)
PRUNER = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1, interval_steps=1)

def construct_model(cls, **kwargs):
    return cls(**kwargs).to(DEVICE)


# Data loaders
def make_loaders(df_tr, df_va, feat_table):
    tr_loader = DataLoader(
        FertilityDataset(df_tr, feat_table),
        batch_size=BATCH_TRAIN,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )
    va_loader = DataLoader(
        FertilityDataset(df_va, feat_table),
        batch_size=BATCH_EVAL,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )
    return tr_loader, va_loader


# Objective functions (prune after each fold)
def objective_mlp(trial, df, feat_table):
    size = trial.suggest_categorical("size", [64, 128, 256, 512, 1024])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.1)
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 3e-3, log=True)

    aucs = []
    for fold in range(N_FOLDS):
        df_va = df[df["fold_id"] == fold].reset_index(drop=True)
        df_tr = df[df["fold_id"] != fold].reset_index(drop=True)

        pos_w = pos_weight_from_df(df_tr)
        tr_loader, va_loader = make_loaders(df_tr, df_va, feat_table)

        model = construct_model(
            BinaryFertilityModel,
            input_dim=feat_table.shape[1],
            size=size,
            dropout_rate=dropout_rate,
        )

        model = train_es_probit_product(
            model, tr_loader, va_loader,
            pos_weight_value=pos_w,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            max_grad_norm=MAX_GRAD_NORM,
            device=DEVICE,
        )

        auc, _ = eval_auc_pr_probit_product(model, va_loader, device=DEVICE)
        aucs.append(auc)

        mean_so_far = float(np.nanmean(aucs))
        trial.report(mean_so_far, step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.nanmean(aucs))


def objective_conv(trial, df, feat_table):
    conv = trial.suggest_categorical("conv", [1, 2, 4, 8, 16])
    pool = trial.suggest_categorical("pool", [512, 1024, 2048, 4096])
    size = trial.suggest_categorical("size", [64, 128, 256, 512, 1024])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.1)
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 3e-3, log=True)

    aucs = []
    for fold in range(N_FOLDS):
        df_va = df[df["fold_id"] == fold].reset_index(drop=True)
        df_tr = df[df["fold_id"] != fold].reset_index(drop=True)

        pos_w = pos_weight_from_df(df_tr)
        tr_loader, va_loader = make_loaders(df_tr, df_va, feat_table)

        model = construct_model(
            BinaryFertilityModelConv,
            input_channels=feat_table.shape[1],
            conv=conv,
            pool=pool,
            size=size,
            dropout_rate=dropout_rate,
        )

        model = train_es_probit_product(
            model, tr_loader, va_loader,
            pos_weight_value=pos_w,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            max_grad_norm=MAX_GRAD_NORM,
            device=DEVICE,
        )

        auc, _ = eval_auc_pr_probit_product(model, va_loader, device=DEVICE)
        aucs.append(auc)

        mean_so_far = float(np.nanmean(aucs))
        trial.report(mean_so_far, step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.nanmean(aucs))


def objective_lasso(trial, df, feat_table):
    l1_lambda = trial.suggest_float("l1_lambda", 1e-4, 10, log=True)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    aucs = []
    for fold in range(N_FOLDS):
        df_va = df[df["fold_id"] == fold].reset_index(drop=True)
        df_tr = df[df["fold_id"] != fold].reset_index(drop=True)

        pos_w = pos_weight_from_df(df_tr)
        tr_loader, va_loader = make_loaders(df_tr, df_va, feat_table)

        model = construct_model(BinaryFertilityLassoTwoTower, input_dim=feat_table.shape[1])

        model = train_es_lasso_probit_product(
            model, tr_loader, va_loader,
            pos_weight_value=pos_w,
            lr=lr,
            l1_lambda=l1_lambda,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            max_grad_norm=MAX_GRAD_NORM,
            device=DEVICE,
        )

        auc, _ = eval_auc_pr_probit_product(model, va_loader, device=DEVICE)
        aucs.append(auc)

        mean_so_far = float(np.nanmean(aucs))
        trial.report(mean_so_far, step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.nanmean(aucs))


# features
A_df = pd.read_csv(FEAT_PATH)
A_df = A_df.rename(columns={A_df.columns[0]: "ID"})
A_df["ID"] = A_df["ID"].astype(str)
feat_table = A_df.set_index("ID").astype(np.float32)

# phenotypes
df = pd.read_csv(PHEN_PATH)

df["BinPheno"] = 1 - df["BinPheno"]
df["BinPheno"] = df["BinPheno"].astype(int)

parts, cache_path = build_cv_folds(
    df,
    splits_dir=SPLITS_DIR,
    n_folds=N_FOLDS,
    seed=SEED,
    tag=TAG,
    )

df = add_fold_id(df, parts)

print(f"N={len(df)} prevalence={1-df['BinPheno'].mean():.3f} device={DEVICE}")

# TT-MLP
study_mlp = optuna.create_study(direction="maximize", sampler=SAMPLER, pruner=PRUNER, study_name=f"MLP_{TAG}")
study_mlp.optimize(lambda t: objective_mlp(t, df, feat_table), n_trials=N_TRIALS_MLP)
print("\n[MLP] best value:", study_mlp.best_value)
print("[MLP] best params:", study_mlp.best_params)

# TT-CNN
study_conv = optuna.create_study(direction="maximize", sampler=SAMPLER, pruner=PRUNER, study_name=f"CONV_{TAG}")
study_conv.optimize(lambda t: objective_conv(t, df, feat_table), n_trials=N_TRIALS_CONV)
print("\n[CONV] best value:", study_conv.best_value)
print("[CONV] best params:", study_conv.best_params)

# TT-LASSO
study_lasso = optuna.create_study(direction="maximize", sampler=SAMPLER, pruner=PRUNER, study_name=f"LASSO_{TAG}")
study_lasso.optimize(lambda t: objective_lasso(t, df, feat_table), n_trials=N_TRIALS_LASSO)
print("\n[LASSO] best value:", study_lasso.best_value)
print("[LASSO] best params:", study_lasso.best_params)

# save summary
rows = [
    {"tag": TAG, "model": "mlp",   "best_value": float(study_mlp.best_value),   **study_mlp.best_params},
    {"tag": TAG, "model": "conv",  "best_value": float(study_conv.best_value),  **study_conv.best_params},
    {"tag": TAG, "model": "lasso", "best_value": float(study_lasso.best_value), **study_lasso.best_params},]

out_df = pd.DataFrame(rows)
out_path = os.path.join(BASE_DIR, f"optuna_best_all3_{TAG}_probit_product.csv")
out_df.to_csv(out_path, index=False)
