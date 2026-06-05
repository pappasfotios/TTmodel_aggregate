from pathlib import Path
import numpy as np
import pandas as pd

scdir = Path("scenarios")

maf_min = 0.12
maf_max = 0.17
qtl_window = 500
rng = np.random.default_rng(123)

for rep in range(1, 6):
    suf = f"HER_M0.4_F0.4_R{rep:02d}_15k.csv"

    feat = pd.read_csv(scdir / f"SimFeat_{suf}", index_col=0)
    phen = pd.read_csv(scdir / f"SimPhen_{suf}")
    qtl = pd.read_csv(scdir / f"SimQTL_{suf}", index_col=0)

    markers = feat.columns.to_list()
    pos = pd.Series({m: int(m.split("_")[1]) for m in markers})

    p = feat.mean(axis=0) / 2.0
    maf = np.minimum(p, 1.0 - p)

    qtl_markers = qtl.iloc[:, 0].astype(str).to_list()
    qtl_pos = [pos[q] for q in qtl_markers if q in pos.index]

    near_qtl = pd.Series(False, index=markers)
    for qp in qtl_pos:
        near_qtl |= (pos >= qp - qtl_window) & (pos <= qp + qtl_window)

    candidates = maf[(maf >= maf_min) & (maf <= maf_max) & (~near_qtl)]

    chosen = rng.choice(candidates.index.to_numpy(), size=2, replace=False)

    fem_marker = chosen[0]
    mal_marker = chosen[1]

    dam_g = phen["Dam"].astype(str).map(feat[fem_marker]).astype(int)
    sire_g = phen["Sire"].astype(str).map(feat[mal_marker]).astype(int)

    #fail = (((dam_g == 2) & (sire_g == 0)) | ((dam_g == 0) & (sire_g == 2)))    #### Failure rule minor homozygotes with major homozygotes
    fail = (((dam_g == 1) & ((sire_g == 0) | (sire_g == 2))) | ((sire_g == 1) & ((dam_g == 0) | (dam_g == 2))))   ### Diamond shape failure

    phen2 = phen.copy()

    phen2["Compat_EggMarker"] = fem_marker
    phen2["Compat_SpermMarker"] = mal_marker
    phen2["Dam_EggGeno"] = dam_g
    phen2["Sire_SpermGeno"] = sire_g
    phen2["CompatFail"] = fail.astype(int)

    phen2["BinPheno_Compat"] = np.where(fail, 1, phen2["BinPheno"].astype(int),)

    phen2["LabelChanged"] = ((phen2["BinPheno"].astype(int) == 0) & fail).astype(int)

    phen2.to_csv(scdir / f"SimPhenCompat_{suf}", index=False)

    summary = pd.DataFrame({
        "stat": [
            "rep",
            "fem_marker",
            "mal_marker",
            "egg_marker_maf",
            "sperm_marker_maf",
            "n_crosses",
            "n_original_failures",
            "n_original_successes",
            "n_compat_fail",
            "compat_fail_rate",
            "n_label_changed_success_to_failure",
            "label_changed_rate",
            "n_final_failures",
            "n_final_successes",
            "final_failure_rate",],

        "value": [
            rep,
            fem_marker,
            mal_marker,
            maf[fem_marker],
            maf[mal_marker],
            len(phen2),
            int((phen2["BinPheno"].astype(int) == 1).sum()),
            int((phen2["BinPheno"].astype(int) == 0).sum()),
            int(fail.sum()),
            float(fail.mean()),
            int(phen2["LabelChanged"].sum()),
            float(phen2["LabelChanged"].mean()),
            int((phen2["BinPheno_Compat"].astype(int) == 1).sum()),
            int((phen2["BinPheno_Compat"].astype(int) == 0).sum()),
            float((phen2["BinPheno_Compat"].astype(int) == 1).mean()),],
            })

    ctab = pd.crosstab(
        dam_g,
        sire_g,
        rownames=["Dam_EggGeno"],
        colnames=["Sire_SpermGeno"],)

    with open(scdir / f"SimCompatSummary_{suf}", "w") as f:
        summary.to_csv(f, index=False)
        f.write("\nGenotype cross-tab counts\n")
        ctab.to_csv(f)

    print(
        rep,
        fem_marker,
        mal_marker,
        "compat_fail:", int(fail.sum()),
        "changed_success_to_failure:", int(phen2["LabelChanged"].sum()),)