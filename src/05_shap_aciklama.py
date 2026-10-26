"""
05_shap_aciklama.py
==================
Girdi:
  - data/processed/train.csv / test.csv
  - data/UCSF-PDGM-metadata.csv
  - models/grade_model.pkl
  - models/idh_model.pkl
  - models/mgmt_model.pkl
  - models/mgmt_selector.pkl
  - models/mgmt_imputer.pkl
  - models/final_columns.pkl

İŞLEM:
  Her görev (Grade, IDH, MGMT) için:
    1. Eğitilmiş model yüklenir
    2. SHAP değerleri hesaplanır (TreeExplainer / KernelExplainer)
    3. Her hasta için en etkili 5 özellik seçilir
    4. Chatbot için JSON-ready CSV kaydedilir (wide format — her hasta 1 satır)
    5. Her hasta için bar plot PNG kaydedilir

ÇIKTI:
  - results/shap_values.csv          (wide format: her hasta 1 satır, görevler sütun olarak)
  - results/shap_plots/<pid>_<gorev>_shap.png  (per-hasta bar plot)
"""

import os, warnings, joblib, json, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
from tqdm import tqdm

warnings.filterwarnings("ignore")

np.random.seed(42)
random.seed(42)

from sklearn.svm            import SVC
from sklearn.impute         import KNNImputer
from xgboost                import XGBClassifier
from lightgbm               import LGBMClassifier

# ─── AYARLAR ───────────────────────────────
TRAIN_PATH    = "data/processed/train.csv"
TEST_PATH     = "data/processed/test.csv"
METADATA_PATH = "data/UCSF-PDGM-metadata.csv"
MODEL_DIR     = "models"
RESULTS_DIR   = "results"
PLOTS_DIR     = os.path.join(RESULTS_DIR, "shap_plots")
TOP_N         = 5

os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════
# VERİ HAZIRLA
# ═══════════════════════════════════════════

def veri_hazirla_shap(rad_df, mgmt_imputer, fit_imputer=False):
    disla    = ["Patient_ID","Grade","Survival Days","1-dead 0-alive",
                "IDH_enc","MGMT_enc","Age At MRI","Sex_enc"]
    rad_cols = [c for c in rad_df.columns if c not in disla]

    rad_df = rad_df.copy()
    rad_df["merge_id"] = (rad_df["Patient_ID"]
                          .str.replace("_nifti","",regex=False)
                          .str.replace("UCSF-PDGM-0","UCSF-PDGM-",regex=False))

    meta = pd.read_csv(METADATA_PATH)
    meta["Yas"]      = pd.to_numeric(meta["Age At MRI"], errors="coerce")
    meta["Cinsiyet"] = (meta["Sex"] == "M").astype(int)

    def idh_enc(v):
        v = str(v).strip().lower()
        if v == "wildtype":           return 1
        if v in ["nan","unknown",""]: return np.nan
        return 0

    meta["IDH_enc"]   = meta["IDH"].apply(idh_enc)
    meta["MGMT_raw"]  = meta["MGMT"].map(
        {"positive":1,"negative":0,"indeterminate":np.nan,"unknown":np.nan})
    meta["Grade_enc"] = pd.to_numeric(meta["Grade"], errors="coerce")

    merged = pd.merge(rad_df,
                      meta[["ID","Yas","Cinsiyet","IDH_enc","MGMT_raw","Grade_enc"]],
                      left_on="merge_id", right_on="ID", how="inner")
    merged = merged.dropna(subset=["Yas","Cinsiyet"]).reset_index(drop=True)

    X  = merged[rad_cols + ["Yas","Cinsiyet"]].copy().reset_index(drop=True)
    et = merged[["Patient_ID","Grade_enc","IDH_enc","MGMT_raw"]].copy().reset_index(drop=True)

    X_vals   = X.values.astype(float)
    mgmt_col = et["MGMT_raw"].values.reshape(-1,1).astype(float)
    X_mgmt   = np.hstack([X_vals, mgmt_col])
    if fit_imputer:
        mgmt_imputer = KNNImputer(n_neighbors=5)
        X_mgmt_imp   = mgmt_imputer.fit_transform(X_mgmt)
    else:
        X_mgmt_imp = mgmt_imputer.transform(X_mgmt)
    et["MGMT_enc"] = np.round(X_mgmt_imp[:,-1]).astype(int)

    return X, et, rad_cols + ["Yas","Cinsiyet"]


# ═══════════════════════════════════════════
# SHAP HESAPLA
# ═══════════════════════════════════════════

def shap_hesapla(model, X_df, model_tipi="tree"):
    X_arr = X_df.values if hasattr(X_df, "values") else X_df

    if model_tipi == "tree":
        from xgboost import XGBClassifier as _XGB
        import json as _json
        
        if isinstance(model, _XGB):
            booster = model.get_booster()
            try:
                cfg = _json.loads(booster.save_config())
                bp = (cfg.get("learner",{})
                         .get("learner_model_param",{})
                         .get("base_score", None))
                if bp is not None:
                    bp_clean = str(bp).strip().strip("[]")
                    cfg["learner"]["learner_model_param"]["base_score"] = bp_clean
                    booster.load_config(_json.dumps(cfg))
            except Exception:
                pass
            
            try:
                explainer = shap.TreeExplainer(booster)
            except (ValueError, TypeError):
                np.random.seed(42)
                rng = np.random.RandomState(42)
                idx = rng.choice(len(X_arr), size=min(100, len(X_arr)), replace=False)
                background = X_arr[idx]
                explainer = shap.KernelExplainer(
                    lambda x: model.predict_proba(x)[:,1], background)
                np.random.seed(42)
            
            shap_vals = explainer.shap_values(X_arr, nsamples=500)
            return shap_vals, float(explainer.expected_value)
        else:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_arr)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            ev = explainer.expected_value
            return shap_vals, float(ev[1] if hasattr(ev, "__len__") else ev)

    elif model_tipi == "kernel":
        np.random.seed(42)
        rng        = np.random.RandomState(42)
        idx        = rng.choice(len(X_arr), size=min(100, len(X_arr)), replace=False)
        background = X_arr[idx]
        explainer  = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:,1], background)
        np.random.seed(42)
        shap_vals  = explainer.shap_values(X_arr, nsamples=500)
        return shap_vals, float(explainer.expected_value)

    elif model_tipi == "linear":
        explainer = shap.LinearExplainer(model, X_arr)
        shap_vals = explainer.shap_values(X_arr)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        return shap_vals, float(explainer.expected_value
                                if not hasattr(explainer.expected_value,"__len__")
                                else explainer.expected_value[1])


# ═══════════════════════════════════════════
# BAR PLOT — tek hasta
# ═══════════════════════════════════════════

def bar_plot_kaydet(pid, gorev, feature_names, shap_row,
                    tahmin_lbl, gercek_lbl, auc_skoru, save_path):
    # Top5: unique features (aynı özellik birden fazla gelmesin, en yüksek SHAP'ı al)
    idx_sorted_all = np.argsort(np.abs(shap_row))[::-1]
    
    idx_sorted = []
    seen_features = set()
    
    for idx in idx_sorted_all:
        fname = feature_names[idx]
        # "firstorder_Skewness_42" → "firstorder_Skewness" (base isim)
        base_name = fname.rsplit("_", 1)[0] if "_" in fname else fname
        
        if base_name not in seen_features:
            idx_sorted.append(idx)
            seen_features.add(base_name)
        
        if len(idx_sorted) == TOP_N:
            break
    
    vals  = shap_row[idx_sorted]
    names = [feature_names[i] for i in idx_sorted]

    def kisalt(name):
        # Index'i ayırma (JSON'da kalır, bar plot'ta yazılmaz)
        # "DWI_log-sigma-1-0-mm-3D_firstorder_Skewness_42"
        # → "DWI_FirstOrder_Skewness"
        
        parts = name.rsplit("_", 1)
        
        if len(parts) == 2 and parts[1].isdigit():
            full_name = parts[0]
            # idx = parts[1]  # Index JSON'da kalır
        else:
            full_name = name
        
        name_parts = full_name.split("_")
        sekans = name_parts[0]
        ozellik = name_parts[-1]
        
        # Yöntem'i bul (firstorder, glcm, lbp, vb.)
        method_map = {
            "firstorder": "FirstOrder",
            "glcm": "GLCM",
            "glrlm": "GLRLM",
            "glszm": "GLSZM",
            "ngtdm": "NGTDM",
            "lbp": "LBP",
            "shape": "Shape",
        }
        
        method_display = None
        for key, val in method_map.items():
            if key in name_parts:
                method_display = val
                break
        
        if method_display:
            return f"{sekans}_{method_display}_{ozellik}"
        else:
            return f"{sekans}_{ozellik}"

    names_short = [kisalt(n) for n in names]
    colors = ["#d73027" if v > 0 else "#4575b4" for v in vals]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(range(len(vals)), vals, color=colors, edgecolor="white", height=0.6)

    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names_short, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP Değeri  (etkinin yönü ve büyüklüğü)", fontsize=9)

    for bar, val in zip(bars, vals):
        xpos = val + 0.002 * (max(abs(vals)) or 1) * (1 if val >= 0 else -1)
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                f"{val:+.3f}", va="center", ha="left" if val >= 0 else "right",
                fontsize=8)

    title = (f"{pid}\n"
             f"Görev: {gorev}  |  Tahmin: {tahmin_lbl}  |  Gerçek: {gercek_lbl}")
    if auc_skoru:
        title += f"  |  AUC: {auc_skoru:.3f}"
    ax.set_title(title, fontsize=9, pad=10)

    red_patch  = mpatches.Patch(color="#d73027", label="Riski / Sınıfı Artırıyor")
    blue_patch = mpatches.Patch(color="#4575b4", label="Riski / Sınıfı Azaltıyor")
    ax.legend(handles=[red_patch, blue_patch], fontsize=8,
              loc="lower right", framealpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════
# SHAP CSV SATIRI OLUSTUR
# ═══════════════════════════════════════════

def shap_csv_satiri(pid, gorev, feature_names, shap_row, tahmin_lbl, gercek_lbl):
    # Top5: unique features (aynı özellik birden fazla gelmesin, en yüksek SHAP'ı al)
    idx_sorted_all = np.argsort(np.abs(shap_row))[::-1]
    
    idx_sorted = []
    seen_features = set()
    
    for idx in idx_sorted_all:
        fname = feature_names[idx]
        # "firstorder_Skewness_42" → "firstorder_Skewness" (base isim)
        base_name = fname.rsplit("_", 1)[0] if "_" in fname else fname
        
        if base_name not in seen_features:
            idx_sorted.append(idx)
            seen_features.add(base_name)
        
        if len(idx_sorted) == TOP_N:
            break
    
    top5_json  = json.dumps(
        {feature_names[i]: round(float(shap_row[i]), 4) for i in idx_sorted},
        ensure_ascii=False)

    row = {
        "Patient_ID": pid,
        "Gorev":      gorev,
        "Tahmin":     tahmin_lbl,
        "Gercek":     gercek_lbl,
        "Top5_JSON":  top5_json,
    }
    for fn, sv in zip(feature_names, shap_row):
        row[f"shap_{fn}"] = round(float(sv), 5)

    return row


# ═══════════════════════════════════════════
# ETİKET ÇEVİRİCİLER
# ═══════════════════════════════════════════

def grade_lbl(v):
    try:
        if np.isnan(float(v)): return "Bilinmiyor"
        return "LowGrade" if int(float(v)) == 0 else "HighGrade"
    except: return "Bilinmiyor"

def idh_lbl(v):
    try:
        if np.isnan(float(v)): return "Bilinmiyor"
        return "Wildtype" if int(float(v)) == 1 else "Mutant"
    except: return "Bilinmiyor"

def mgmt_lbl(v):
    try:
        if np.isnan(float(v)): return "Bilinmiyor"
        return "Metile" if int(float(v)) == 1 else "Metilsiz"
    except: return "Bilinmiyor"


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def main():
    print("\n" + "="*65)
    print("05_shap_analizi.py  basliyor...")
    print(f"  TOP_N={TOP_N}  |  Tum hastalar (train+test)  |  seed=42  |  nsamples=500")
    print("="*65)

    model_grade   = joblib.load(os.path.join(MODEL_DIR, "grade_model.pkl"))
    model_idh     = joblib.load(os.path.join(MODEL_DIR, "idh_model.pkl"))
    model_mgmt    = joblib.load(os.path.join(MODEL_DIR, "mgmt_model.pkl"))
    selector_mgmt = joblib.load(os.path.join(MODEL_DIR, "mgmt_selector.pkl"))
    imputer_mgmt  = joblib.load(os.path.join(MODEL_DIR, "mgmt_imputer.pkl"))

    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    X_tr, et_tr, feat_names = veri_hazirla_shap(
        train_df, mgmt_imputer=imputer_mgmt, fit_imputer=False)
    X_te, et_te, _          = veri_hazirla_shap(
        test_df,  mgmt_imputer=imputer_mgmt, fit_imputer=False)

    X_all  = pd.concat([X_tr, X_te], ignore_index=True)
    et_all = pd.concat([et_tr, et_te], ignore_index=True)
    n_all  = len(et_all)
    print(f"  Toplam hasta: {n_all}  (Train:{len(et_tr)}  Test:{len(et_te)})")

    all_rows = []

    # ════════════════════════════════════════
    # GRADE — SVM → KernelExplainer
    # ════════════════════════════════════════
    print("\n--- GRADE SHAP (SVM / KernelExplainer) ---")
    print("    [Bu adım biraz sürebilir...]")

    grade_mask = et_all["Grade_enc"].notna()
    X_g        = X_all[grade_mask].values
    et_g       = et_all[grade_mask].reset_index(drop=True)
    ids_g      = et_g["Patient_ID"].values

    y_g_enc = np.where(et_g["Grade_enc"].astype(int).values == 4, 1, 0)
    proba_g = model_grade.predict_proba(X_g)
    ypred_g = (proba_g[:,1] >= 0.40).astype(int)

    shap_g, _ = shap_hesapla(model_grade, X_g, model_tipi="kernel")
    print(f"  SHAP hesaplandi: {shap_g.shape}")

    for i, pid in enumerate(ids_g):
        tahmin = grade_lbl(ypred_g[i])
        gercek = "LowGrade" if y_g_enc[i] == 0 else "HighGrade"
        save_p = os.path.join(PLOTS_DIR, f"{pid}_Grade_shap.png")
        bar_plot_kaydet(pid, "Grade", feat_names, shap_g[i],
                        tahmin, gercek, None, save_p)
        all_rows.append(shap_csv_satiri(pid, "Grade", feat_names, shap_g[i],
                                        tahmin, gercek))

    print(f"  {len(ids_g)} hasta tamamlandi → {PLOTS_DIR}/")

    # ════════════════════════════════════════
    # IDH — XGBoost → TreeExplainer
    # ════════════════════════════════════════
    print("\n--- IDH SHAP (XGBoost / TreeExplainer) ---")

    idh_mask = et_all["IDH_enc"].isin([0,1])
    X_i      = X_all[idh_mask].values
    et_i     = et_all[idh_mask].reset_index(drop=True)
    ids_i    = et_i["Patient_ID"].values

    y_i_enc = et_i["IDH_enc"].astype(int).values
    proba_i = model_idh.predict_proba(X_i)
    ypred_i = (proba_i[:,1] >= 0.40).astype(int)

    shap_i, _ = shap_hesapla(model_idh, X_i, model_tipi="tree")
    print(f"  SHAP hesaplandi: {shap_i.shape}")

    for i, pid in enumerate(ids_i):
        tahmin = idh_lbl(ypred_i[i])
        gercek = idh_lbl(y_i_enc[i])
        save_p = os.path.join(PLOTS_DIR, f"{pid}_IDH_shap.png")
        bar_plot_kaydet(pid, "IDH", feat_names, shap_i[i],
                        tahmin, gercek, None, save_p)
        all_rows.append(shap_csv_satiri(pid, "IDH", feat_names, shap_i[i],
                                        tahmin, gercek))

    print(f"  {len(ids_i)} hasta tamamlandi → {PLOTS_DIR}/")

    # ════════════════════════════════════════
    # MGMT — LightGBM → TreeExplainer
    # ════════════════════════════════════════
    print("\n--- MGMT SHAP (LightGBM / TreeExplainer) ---")

    X_m_all      = selector_mgmt.transform(X_all.values)
    y_m_enc      = et_all["MGMT_enc"].values.astype(int)
    ids_m        = et_all["Patient_ID"].values
    sel_mask     = selector_mgmt.get_support()
    feat_names_m = [f"{feat_names[i]}_{i}" for i, s in enumerate(sel_mask) if s]

    proba_m = model_mgmt.predict_proba(X_m_all)
    ypred_m = (proba_m[:,1] >= 0.50).astype(int)

    shap_m, _ = shap_hesapla(model_mgmt, X_m_all, model_tipi="tree")
    shap_m = shap_m * -1

    print(f"  SHAP hesaplandi: {shap_m.shape}")

    for i, pid in enumerate(tqdm(ids_m, ncols=110)):
        tahmin = mgmt_lbl(ypred_m[i])
        gercek = mgmt_lbl(y_m_enc[i])
        save_p = os.path.join(PLOTS_DIR, f"{pid}_MGMT_shap.png")
        bar_plot_kaydet(pid, "MGMT", feat_names_m, shap_m[i],
                        tahmin, gercek, None, save_p)
        all_rows.append(shap_csv_satiri(pid, "MGMT", feat_names_m, shap_m[i],
                                        tahmin, gercek))

    print(f"  {len(ids_m)} hasta tamamlandi → {PLOTS_DIR}/")

    # ════════════════════════════════════════
    # SHAP VALUES CSV KAYDET  (wide format)
    # ════════════════════════════════════════
    print("\n--- CSV kaydediliyor ---")

    df_long  = pd.DataFrame(all_rows)
    gorevler = ["Grade", "IDH", "MGMT"]
    df_wide  = None

    for gorev in gorevler:
        df_g = df_long[df_long["Gorev"] == gorev][
            ["Patient_ID", "Tahmin", "Gercek", "Top5_JSON"]
        ].copy()
        df_g = df_g.rename(columns={
            "Tahmin":    f"{gorev}_Tahmin",
            "Gercek":    f"{gorev}_Gercek",
            "Top5_JSON": f"{gorev}_Top5_JSON",
        })
        df_wide = df_g if df_wide is None else pd.merge(df_wide, df_g, on="Patient_ID", how="outer")

    col_order = ["Patient_ID"]
    for gorev in gorevler:
        col_order += [f"{gorev}_Tahmin", f"{gorev}_Gercek", f"{gorev}_Top5_JSON"]
    df_wide = df_wide[col_order]

    shap_chat_p = os.path.join(RESULTS_DIR, "shap_values.csv")
    df_wide.to_csv(shap_chat_p, index=False, encoding="utf-8-sig", sep=";", decimal=",")

    print(f"\n  shap_values.csv (chatbot, wide) : {shap_chat_p}")
    print(f"  Bar plot PNG sayisi             : {len(all_rows)}")
    print(f"  Plot klasoru                    : {PLOTS_DIR}/")

    print("\n" + "="*65)
    print("TAMAMLANDI")
    print("="*65)


if __name__ == "__main__":
    main()