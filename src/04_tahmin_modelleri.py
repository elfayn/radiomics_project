"""
04_tahmin_modelleri.py
======================
Girdi:
  - data/processed/temizlenmis.csv
  - data/UCSF-PDGM-metadata.csv

KAZANAN MODELLER:
  Grade (LowGrade=II+III  vs  HighGrade=IV) : SVM
  IDH   (Mutant vs Wildtype)                : XGBoost
  MGMT  (Metilsiz vs Metile)               : LightGBM

MGMT AKIŞI:
  1. Tüm hastalar alınır (positive/negative/indeterminate/unknown)
  2. indeterminate + unknown → KNN imputer ile doldurulur (eksik veri muamelesi)
  3. Dengelenmiş tam set → SMOTE ile artırılır
  4. Model eğitilir

CSV SUTUN YAPISI (her hasta = 1 satir):
  Patient_ID |
  LowGrade_oran | HighGrade_oran | Grade_tahmin |
  IDH_mutant_oran | IDH_wildtype_oran | IDH_tahmin |
  MGMT_metilsiz_oran | MGMT_metile_oran | MGMT_tahmin |
  Gercek_Grade | Gercek_ID | Gercek_MGMT

CSV FORMAT: noktalı virgül ayırıcı (;), ondalık virgül (,) — Türkçe Excel uyumlu

CIKTI:
  - data/processed/train.csv / test.csv
  - models/*.pkl
  - results/tahmin_sonuclari.csv
  - results/bagimsiz_test_tahminleri.csv
"""

import os, warnings, joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.svm               import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute            import KNNImputer
from sklearn.model_selection   import train_test_split
from sklearn.metrics           import (f1_score, roc_auc_score,
                                        accuracy_score,
                                        precision_recall_fscore_support)
from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

# ─── AYARLAR ───────────────────────────────
RADIOMICS_PATH = "data/processed/temizlenmis.csv"
TRAIN_PATH     = "data/processed/train.csv"
TEST_PATH      = "data/processed/test.csv"
METADATA_PATH  = "data/UCSF-PDGM-metadata.csv"
MODEL_DIR      = "models"
RESULTS_DIR    = "results"
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
MGMT_K         = 15

os.makedirs(MODEL_DIR,        exist_ok=True)
os.makedirs(RESULTS_DIR,      exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


# ═══════════════════════════════════════════
# ADIM 0 — GLOBAL SPLIT
# ═══════════════════════════════════════════

def train_test_bol():
    if os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH):
        tr = pd.read_csv(TRAIN_PATH)
        te = pd.read_csv(TEST_PATH)
        print(f"Mevcut split kullaniliyor → Train:{len(tr)}  Test:{len(te)}")
        return tr, te

    print("Global 80/20 split yapiliyor (ilk kez)...")
    rad  = pd.read_csv(RADIOMICS_PATH)
    meta = pd.read_csv(METADATA_PATH)
    meta["Grade_enc"] = pd.to_numeric(meta["Grade"], errors="coerce")
    rad["merge_id"] = (rad["Patient_ID"]
                       .str.replace("_nifti", "", regex=False)
                       .str.replace("UCSF-PDGM-0", "UCSF-PDGM-", regex=False))
    merged = pd.merge(rad, meta[["ID","Grade_enc"]],
                      left_on="merge_id", right_on="ID", how="inner")
    merged = merged.dropna(subset=["Grade_enc"]).reset_index(drop=True)
    idx_tr, idx_te = train_test_split(
        merged.index, test_size=TEST_SIZE,
        stratify=merged["Grade_enc"], random_state=RANDOM_STATE)
    rad_c  = rad.drop(columns=["merge_id"])
    tr_ids = merged.loc[idx_tr, "Patient_ID"].values
    te_ids = merged.loc[idx_te, "Patient_ID"].values
    tr = rad_c[rad_c["Patient_ID"].isin(tr_ids)].reset_index(drop=True)
    te = rad_c[rad_c["Patient_ID"].isin(te_ids)].reset_index(drop=True)
    tr.to_csv(TRAIN_PATH, index=False)
    te.to_csv(TEST_PATH,  index=False)
    print(f"  Train:{len(tr)}  Test:{len(te)}")
    return tr, te


# ═══════════════════════════════════════════
# ADIM 1 — VERİ HAZIRLA
# MGMT: positive/negative/indeterminate/unknown hepsi alınır
#        indeterminate+unknown → NaN olarak işaretlenir, KNN ile doldurulur
# ═══════════════════════════════════════════

def veri_hazirla(rad_df, mgmt_imputer=None, fit_imputer=False, split_adi="Train"):
    disla    = ["Patient_ID","Grade","Survival Days","1-dead 0-alive",
                "IDH_enc","MGMT_enc","Age At MRI","Sex_enc"]
    rad_cols = [c for c in rad_df.columns if c not in disla]

    rad_df = rad_df.copy()
    rad_df["merge_id"] = (rad_df["Patient_ID"]
                          .str.replace("_nifti", "", regex=False)
                          .str.replace("UCSF-PDGM-0", "UCSF-PDGM-", regex=False))

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
        {"positive": 1, "negative": 0,
         "indeterminate": np.nan, "unknown": np.nan})
    meta["Grade_enc"] = pd.to_numeric(meta["Grade"], errors="coerce")

    merged = pd.merge(rad_df,
                      meta[["ID","Yas","Cinsiyet","IDH_enc","MGMT_raw","Grade_enc"]],
                      left_on="merge_id", right_on="ID", how="inner")
    merged = merged.dropna(subset=["Yas","Cinsiyet"]).reset_index(drop=True)

    X  = merged[rad_cols + ["Yas","Cinsiyet"]].copy().reset_index(drop=True)
    et = merged[["Patient_ID","Grade_enc","IDH_enc","MGMT_raw"]].copy().reset_index(drop=True)

    # ── MGMT KNN IMPUTATION ──────────────────────────────────────
    mgmt_nan_count = et["MGMT_raw"].isna().sum()
    print(f"  [{split_adi}] MGMT eksik (indeterminate/unknown): {mgmt_nan_count} hasta → KNN ile doldurulacak")

    X_vals   = X.values.astype(float)
    mgmt_col = et["MGMT_raw"].values.reshape(-1, 1).astype(float)
    X_mgmt   = np.hstack([X_vals, mgmt_col])

    if fit_imputer:
        mgmt_imputer = KNNImputer(n_neighbors=5)
        X_mgmt_imp   = mgmt_imputer.fit_transform(X_mgmt)
        joblib.dump(mgmt_imputer, os.path.join(MODEL_DIR, "mgmt_imputer.pkl"))
    else:
        X_mgmt_imp = mgmt_imputer.transform(X_mgmt)

    et["MGMT_enc"] = np.round(X_mgmt_imp[:, -1]).astype(int)

    if mgmt_nan_count > 0:
        dolu = et["MGMT_enc"].value_counts().sort_index()
        print(f"  [{split_adi}] KNN sonrasi MGMT → Metilsiz(0):{dolu.get(0,0)}  Metile(1):{dolu.get(1,0)}")

    return X, et, rad_cols, mgmt_imputer


# ─── YARDIMCI FONKSİYONLAR ─────────────────

def smote_dagil_goster(y_once, y_sonra, yontem, sinif_isimleri):
    """SMOTE öncesi ve sonrası sınıf dağılımını tablolu gösterir."""
    sayim_once  = pd.Series(y_once).value_counts().sort_index()
    sayim_sonra = pd.Series(y_sonra).value_counts().sort_index()
    # En uzun sınıf adına göre kolon genişliğini otomatik ayarla
    max_len = max(len(v) for v in sinif_isimleri.values())
    max_len = max(max_len, len("TOPLAM"), 6)
    w = max_len + 2
    print(f"\n  [{yontem}] Veri arttirim oncesi → sonrasi:")
    print(f"  {'Sinif':>{w}} {'Once':>8} {'Sonra':>8} {'Fark':>8}")
    print(f"  {'-'*(w+26)}")
    for enc, isim in sinif_isimleri.items():
        once  = sayim_once.get(enc, 0)
        sonra = sayim_sonra.get(enc, 0)
        print(f"  {isim:>{w}} {once:>8} {sonra:>8} {sonra-once:>+8}")
    print(f"  {'-'*(w+26)}")
    print(f"  {'TOPLAM':>{w}} {len(y_once):>8} {len(y_sonra):>8} {len(y_sonra)-len(y_once):>+8}")


def smote_uygula(X, y, hedef):
    m = {int(k):int(v) for k,v in pd.Series(y).value_counts().items()}
    s = {c:h for c,h in hedef.items() if h > m.get(c,0)}
    if not s: return X, y
    k = max(1, min(5, min(m.get(c,0) for c in s)-1))
    return SMOTE(sampling_strategy=s, k_neighbors=k,
                 random_state=RANDOM_STATE).fit_resample(X, y)

def bsmote_uygula(X, y, hedef):
    m = {int(k):int(v) for k,v in pd.Series(y).value_counts().items()}
    s = {c:h for c,h in hedef.items() if h > m.get(c,0)}
    if not s: return X, y
    k = max(1, min(5, min(m.get(c,0) for c in s)-1))
    return BorderlineSMOTE(sampling_strategy=s, k_neighbors=k,
                           kind="borderline-1",
                           random_state=RANDOM_STATE).fit_resample(X, y)

def rapor_yazdir(y_te, y_pred, sinif_etiketleri, sinif_sayilari, model_isim, auc=None):
    pres, recs, f1s, sups = precision_recall_fscore_support(
        y_te, y_pred, labels=sinif_sayilari, zero_division=0)
    acc    = accuracy_score(y_te, y_pred)
    mac_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
    mac_p  = float(np.mean(pres))
    mac_r  = float(np.mean(recs))

    print(f"\n  [{model_isim}] TEST SETI raporu ({len(y_te)} hasta):")
    print(f"  {'':20s} {'precision':>10} {'recall':>8} {'f1-score':>9} {'support':>8}")
    print(f"  {'-'*57}")
    for i, isim_s in enumerate(sinif_etiketleri):
        print(f"  {isim_s:20s} {pres[i]:>10.2f} {recs[i]:>8.2f} {f1s[i]:>9.2f} {sups[i]:>8}")
    print(f"  {'-'*57}")
    print(f"  {'accuracy':20s} {'':>10} {'':>8} {acc:>9.2f} {len(y_te):>8}")
    print(f"  {'macro avg':20s} {mac_p:>10.2f} {mac_r:>8.2f} {mac_f1:>9.2f} {len(y_te):>8}")
    if auc is not None:
        print(f"  {'AUC-ROC':20s} {'':>10} {'':>8} {auc:>9.4f}")


def doldur_gorev(mask_arr, proba, y_pred, y_true_enc, n):
    """Mask'li tahminleri tüm hasta dizisine yay (NaN ile)."""
    proba_full = np.full((n, 2), np.nan)
    ypred_full = np.full(n, np.nan)
    ytrue_full = np.full(n, np.nan)
    j = 0
    for i in range(n):
        if mask_arr[i]:
            proba_full[i]  = proba[j]
            ypred_full[i]  = y_pred[j]
            ytrue_full[i]  = y_true_enc[j]
            j += 1
    return proba_full, ypred_full, ytrue_full


# ═══════════════════════════════════════════
# WIDE FORMAT CSV BUILDER
# ═══════════════════════════════════════════

def _pct(v):
    """float → yüzde, 1 ondalık, NaN korumalı."""
    try:
        if np.isnan(float(v)): return np.nan
        return round(float(v) * 100, 1)
    except:
        return np.nan

def wide_df_olustur(
    ids,
    grade_raw,
    proba_grade_full,
    ypred_grade_full,
    idh_mask_arr,
    proba_idh_full,
    ypred_idh_full,
    ytrue_idh_full,
    proba_mgmt_full,
    ypred_mgmt_full,
    ytrue_mgmt_full,
    mgmt_original_vals,
):
    n = len(ids)

    def _grade_gercek(v):
        try:
            if np.isnan(float(v)): return ""
            return int(float(v))
        except: return ""

    def _grade_tahmin(v):
        try:
            if np.isnan(float(v)): return ""
            return "LowGrade" if int(float(v)) == 0 else "HighGrade"
        except: return ""

    def _idh_gercek(v):
        try:
            if np.isnan(float(v)): return "Bilinmiyor"
            return "Wildtype" if int(float(v)) == 1 else "Mutant"
        except: return "Bilinmiyor"

    def _idh_tahmin(v):
        try:
            if np.isnan(float(v)): return ""
            return "Wildtype" if int(float(v)) == 1 else "Mutant"
        except: return ""

    def _mgmt_gercek(orig_val):
        try:
            if np.isnan(float(orig_val)): return "Bilinmiyor"
            return "Metile" if int(float(orig_val)) == 1 else "Metilsiz"
        except: return "Bilinmiyor"

    def _mgmt_tahmin(v):
        try:
            if np.isnan(float(v)): return ""
            return "Metile" if int(float(v)) == 1 else "Metilsiz"
        except: return ""

    rows = []
    for i in range(n):
        row = {"Patient_ID": ids[i]}

        # ── GRADE ────────────────────────────────────────────────────
        # LowGrade_oran  = Grade II + III birlesik olasiligi (%)
        # HighGrade_oran = Grade IV olasiligi (%)
        row["LowGrade_oran"]  = _pct(proba_grade_full[i, 0])
        row["HighGrade_oran"] = _pct(proba_grade_full[i, 1])
        row["Grade_tahmin"]   = _grade_tahmin(ypred_grade_full[i])

        # ── IDH ──────────────────────────────────────────────────────
        if idh_mask_arr[i]:
            row["IDH_mutant_oran"]   = _pct(proba_idh_full[i, 0])
            row["IDH_wildtype_oran"] = _pct(proba_idh_full[i, 1])
            row["IDH_tahmin"]        = _idh_tahmin(ypred_idh_full[i])
        else:
            row["IDH_mutant_oran"]   = np.nan
            row["IDH_wildtype_oran"] = np.nan
            row["IDH_tahmin"]        = ""

        # ── MGMT ─────────────────────────────────────────────────────
        row["MGMT_metilsiz_oran"] = _pct(proba_mgmt_full[i, 0])
        row["MGMT_metile_oran"]   = _pct(proba_mgmt_full[i, 1])
        row["MGMT_tahmin"]        = _mgmt_tahmin(ypred_mgmt_full[i])

        # ── GERÇEK ETİKETLER ─────────────────────────────────────────
        row["Gercek_Grade"] = _grade_gercek(grade_raw[i])
        row["Gercek_IDH"]   = _idh_gercek(ytrue_idh_full[i])
        row["Gercek_MGMT"]  = _mgmt_gercek(mgmt_original_vals[i])

        rows.append(row)

    col_order = [
        "Patient_ID",
        "LowGrade_oran", "HighGrade_oran", "Grade_tahmin",
        "IDH_mutant_oran", "IDH_wildtype_oran", "IDH_tahmin",
        "MGMT_metilsiz_oran", "MGMT_metile_oran", "MGMT_tahmin",
        "Gercek_Grade", "Gercek_IDH", "Gercek_MGMT",
    ]
    return pd.DataFrame(rows, columns=col_order)


def kaydet_csv(df, path):
    """Türkçe Excel uyumlu CSV: noktalı virgül ayırıcı, ondalık virgül."""
    df.to_csv(path, index=False, encoding="utf-8-sig", sep=";", decimal=",")


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def main():
    print("\n" + "="*65)
    print("04_tahmin_modelleri.py  basliyor...")
    print("  Grade=SVM | IDH=XGBoost | MGMT=LightGBM")
    print("  MGMT: KNN impute (indeterminate/unknown) → SMOTE")
    print("="*65)

    train_df, test_df = train_test_bol()

    print("\n--- Veri Hazirlama (MGMT KNN imputation dahil) ---")
    X_tr, et_tr, rad_cols, mgmt_imputer = veri_hazirla(train_df, fit_imputer=True,  split_adi="Train")
    X_te, et_te, _, _                   = veri_hazirla(test_df,  mgmt_imputer=mgmt_imputer, fit_imputer=False, split_adi="Test")
    joblib.dump(rad_cols, os.path.join(MODEL_DIR, "final_columns.pkl"))
    print(f"  Train:{X_tr.shape}  Test:{X_te.shape}")

    n_tr = len(et_tr)
    n_te = len(et_te)

    # Orijinal MGMT değerlerini sakla (Gercek_MGMT için, impute öncesi)
    meta_raw = pd.read_csv(METADATA_PATH)
    meta_raw["MGMT_orig"] = meta_raw["MGMT"].map(
        {"positive": 1.0, "negative": 0.0,
         "indeterminate": np.nan, "unknown": np.nan})

    def align_orig(et_df, rad_df):
        rad_df = rad_df.copy()
        rad_df["merge_id"] = (rad_df["Patient_ID"]
                              .str.replace("_nifti","",regex=False)
                              .str.replace("UCSF-PDGM-0","UCSF-PDGM-",regex=False))
        merged2 = pd.merge(et_df[["Patient_ID"]],
                           pd.merge(rad_df[["Patient_ID","merge_id"]],
                                    meta_raw[["ID","MGMT_orig"]],
                                    left_on="merge_id", right_on="ID", how="left"),
                           on="Patient_ID", how="left")
        return merged2["MGMT_orig"].values

    mgmt_orig_tr = align_orig(et_tr, train_df)
    mgmt_orig_te = align_orig(et_te, test_df)

    # ════════════════════════════════
    # GRADE — SVM
    # ════════════════════════════════
    print("\n" + "="*65)
    print("GRADE  (LowGrade=II+III  vs  HighGrade=IV)  —  SVM")
    print("="*65)

    grade_mask_tr = et_tr["Grade_enc"].notna()
    grade_mask_te = et_te["Grade_enc"].notna()
    grade_raw_tr  = et_tr["Grade_enc"].values
    grade_raw_te  = et_te["Grade_enc"].values

    X_g_tr = X_tr[grade_mask_tr].values
    X_g_te = X_te[grade_mask_te].values
    y_g_tr = np.where(et_tr.loc[grade_mask_tr,"Grade_enc"].astype(int).values == 4, 1, 0)
    y_g_te = np.where(et_te.loc[grade_mask_te,"Grade_enc"].astype(int).values == 4, 1, 0)

    print(f"  Train: LowGrade={int((y_g_tr==0).sum())}  HighGrade={int((y_g_tr==1).sum())}")
    print(f"  Test : LowGrade={int((y_g_te==0).sum())}  HighGrade={int((y_g_te==1).sum())}")

    X_g_res, y_g_res = bsmote_uygula(
        X_g_tr, y_g_tr, {0: max(int((y_g_tr==0).sum()), 150)})
    smote_dagil_goster(y_g_tr, y_g_res, "BSMOTE",
                       {0:"LowGrade(II+III)", 1:"HighGrade(IV)"})

    model_grade = SVC(kernel="rbf", class_weight={0:3, 1:1},
                      probability=True, random_state=RANDOM_STATE)
    model_grade.fit(X_g_res, y_g_res)
    joblib.dump(model_grade, os.path.join(MODEL_DIR, "grade_model.pkl"))

    proba_g_te  = model_grade.predict_proba(X_g_te)
    y_pred_g_te = (proba_g_te[:,1] >= 0.40).astype(int)
    auc_g = round(roc_auc_score(y_g_te, proba_g_te[:,1]), 4)
    rapor_yazdir(y_g_te, y_pred_g_te, ["LowGrade","HighGrade"], [0,1], "SVM (Grade)", auc_g)

    proba_g_tr  = model_grade.predict_proba(X_g_tr)
    y_pred_g_tr = (proba_g_tr[:,1] >= 0.40).astype(int)

    proba_g_tr_full, ypred_g_tr_full, _ = doldur_gorev(
        grade_mask_tr.values, proba_g_tr, y_pred_g_tr, y_g_tr, n_tr)
    proba_g_te_full, ypred_g_te_full, _ = doldur_gorev(
        grade_mask_te.values, proba_g_te, y_pred_g_te, y_g_te, n_te)

    # ════════════════════════════════
    # IDH — XGBoost
    # ════════════════════════════════
    print("\n" + "="*65)
    print("IDH  (0=Mutant / 1=Wildtype)  —  XGBoost")
    print("="*65)

    idh_mask_tr = et_tr["IDH_enc"].isin([0,1])
    idh_mask_te = et_te["IDH_enc"].isin([0,1])
    X_i_tr = X_tr[idh_mask_tr].values
    X_i_te = X_te[idh_mask_te].values
    y_i_tr = et_tr.loc[idh_mask_tr,"IDH_enc"].astype(int).values
    y_i_te = et_te.loc[idh_mask_te,"IDH_enc"].astype(int).values

    print(f"  Train: Mutant={int((y_i_tr==0).sum())}  Wildtype={int((y_i_tr==1).sum())}")
    print(f"  Test : Mutant={int((y_i_te==0).sum())}  Wildtype={int((y_i_te==1).sum())}")

    spw_i = round(int((y_i_tr==1).sum()) / max(int((y_i_tr==0).sum()),1), 2)
    X_i_res, y_i_res = smote_uygula(X_i_tr, y_i_tr, {0: 185})
    smote_dagil_goster(y_i_tr, y_i_res, "SMOTE",
                       {0:"Mutant", 1:"Wildtype"})

    model_idh = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=spw_i, eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1)
    model_idh.fit(X_i_res, y_i_res)
    joblib.dump(model_idh, os.path.join(MODEL_DIR, "idh_model.pkl"))

    proba_i_te  = model_idh.predict_proba(X_i_te)
    y_pred_i_te = (proba_i_te[:,1] >= 0.40).astype(int)
    auc_i = round(roc_auc_score(y_i_te, proba_i_te[:,1]), 4)
    rapor_yazdir(y_i_te, y_pred_i_te, ["Mutant","Wildtype"], [0,1], "XGBoost (IDH)", auc_i)

    proba_i_tr  = model_idh.predict_proba(X_i_tr)
    y_pred_i_tr = (proba_i_tr[:,1] >= 0.40).astype(int)

    proba_i_tr_full, ypred_i_tr_full, ytrue_i_tr_full = doldur_gorev(
        idh_mask_tr.values, proba_i_tr, y_pred_i_tr, y_i_tr, n_tr)
    proba_i_te_full, ypred_i_te_full, ytrue_i_te_full = doldur_gorev(
        idh_mask_te.values, proba_i_te, y_pred_i_te, y_i_te, n_te)

    # ════════════════════════════════
    # MGMT — LightGBM
    # ════════════════════════════════
    print("\n" + "="*65)
    print("MGMT  (0=Metilsiz / 1=Metile)  —  LightGBM")
    print("  [KNN imputation veri_hazirla'da tamamlandi]")
    print("="*65)

    y_m_tr = et_tr["MGMT_enc"].values.astype(int)
    y_m_te = et_te["MGMT_enc"].values.astype(int)
    X_m_tr = X_tr.values
    X_m_te = X_te.values

    print(f"  Train (KNN sonrasi): Metilsiz={int((y_m_tr==0).sum())}  Metile={int((y_m_tr==1).sum())}")
    print(f"  Test  (KNN sonrasi): Metilsiz={int((y_m_te==0).sum())}  Metile={int((y_m_te==1).sum())}")

    # SelectKBest: k=15 (gürültü azaltmak için)
    selector  = SelectKBest(f_classif, k=15)
    X_m_tr_s  = selector.fit_transform(X_m_tr, y_m_tr)
    X_m_te_s  = selector.transform(X_m_te)
    joblib.dump(selector, os.path.join(MODEL_DIR, "mgmt_selector.pkl"))
    print(f"  SelectKBest: {X_m_tr.shape[1]} → 15 ozellik")

    # SMOTE: azınlık/çoğunluk oranı = 0.70 → Metilsiz hedef = 302*0.70 ≈ 211
    X_m_res, y_m_res = smote_uygula(X_m_tr_s, y_m_tr, {0: 211})
    smote_dagil_goster(y_m_tr, y_m_res, "SMOTE",
                       {0:"Metilsiz", 1:"Metile"})

    model_mgmt = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=15,
        reg_alpha=0.5,
        reg_lambda=1.0,
        class_weight={0:4, 1:1},
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    model_mgmt.fit(X_m_res, y_m_res)
    joblib.dump(model_mgmt, os.path.join(MODEL_DIR, "mgmt_model.pkl"))

    proba_m_te  = model_mgmt.predict_proba(X_m_te_s)
    y_pred_m_te = (proba_m_te[:,1] >= 0.50).astype(int)
    auc_m = round(roc_auc_score(y_m_te, proba_m_te[:,1]), 4)
    rapor_yazdir(y_m_te, y_pred_m_te, ["Metilsiz","Metile"], [0,1], "LightGBM (MGMT)", auc_m)

    proba_m_tr  = model_mgmt.predict_proba(X_m_tr_s)
    y_pred_m_tr = (proba_m_tr[:,1] >= 0.50).astype(int)

    # ════════════════════════════════
    # WIDE FORMAT CSV
    # ════════════════════════════════

    df_tr_wide = wide_df_olustur(
        ids               = et_tr["Patient_ID"].values,
        grade_raw         = grade_raw_tr,
        proba_grade_full  = proba_g_tr_full,
        ypred_grade_full  = ypred_g_tr_full,
        idh_mask_arr      = idh_mask_tr.values,
        proba_idh_full    = proba_i_tr_full,
        ypred_idh_full    = ypred_i_tr_full,
        ytrue_idh_full    = ytrue_i_tr_full,
        proba_mgmt_full   = proba_m_tr,
        ypred_mgmt_full   = y_pred_m_tr,
        ytrue_mgmt_full   = y_m_tr,
        mgmt_original_vals= mgmt_orig_tr,
    )

    df_te_wide = wide_df_olustur(
        ids               = et_te["Patient_ID"].values,
        grade_raw         = grade_raw_te,
        proba_grade_full  = proba_g_te_full,
        ypred_grade_full  = ypred_g_te_full,
        idh_mask_arr      = idh_mask_te.values,
        proba_idh_full    = proba_i_te_full,
        ypred_idh_full    = ypred_i_te_full,
        ytrue_idh_full    = ytrue_i_te_full,
        proba_mgmt_full   = proba_m_te,
        ypred_mgmt_full   = y_pred_m_te,
        ytrue_mgmt_full   = y_m_te,
        mgmt_original_vals= mgmt_orig_te,
    )

    # 1) tahmin_sonuclari.csv — train + test
    tum_wide = pd.concat([df_tr_wide, df_te_wide], ignore_index=True)
    tum_p    = os.path.join(RESULTS_DIR, "tahmin_sonuclari.csv")
    kaydet_csv(tum_wide, tum_p)

    # 2) bagimsiz_test_tahminleri.csv — sadece test
    test_p = os.path.join(RESULTS_DIR, "bagimsiz_test_tahminleri.csv")
    kaydet_csv(df_te_wide, test_p)

    print("\n" + "="*65)
    print("TAMAMLANDI")
    print(f"  tahmin_sonuclari.csv        : {len(tum_wide)} hasta  ({len(df_tr_wide)} train + {len(df_te_wide)} test)")
    print(f"  bagimsiz_test_tahminleri.csv: {len(df_te_wide)} hasta")
    print("  grade_model.pkl | idh_model.pkl | mgmt_model.pkl")
    print("="*65)


if __name__ == "__main__":
    main()