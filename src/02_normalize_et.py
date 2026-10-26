import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

INPUT_PATH = "data/processed/radiomics.csv"
CLINICAL_PATH = "data/UCSF-PDGM-metadata.csv"
OUTPUT_PATH = "data/processed/temizlenmis.csv"

def main():

    # ── RADİOMİCS ──────────────────────────────────────────
    df = pd.read_csv(INPUT_PATH)
    patient_ids = df["Patient_ID"].reset_index(drop=True)
    X = df.drop(columns=["Patient_ID"])

    print(f"Ham ozellik sayisi: {X.shape[1]}")

    X = X.loc[:, ~X.columns.str.startswith('segmentation')]
    print(f"Segmentation temizligi sonrasi: {X.shape[1]} kolon")

    X = X.loc[:, X.isna().mean() < 0.20]
    print(f"NaN filtresi sonrasi: {X.shape[1]} kolon")

    # Kompleks sayi temizligi
    def temizle(val):
        try:
            return float(complex(val).real)
        except:
            return np.nan

    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].apply(temizle)
            X[col] = pd.to_numeric(X[col], errors='coerce')

    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    print("NaN doldurma tamamlandi")

    # Z-SCORE NORMALIZATION
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    joblib.dump(scaler, "models/scaler.pkl")

    # VARIANCE FILTER
    selector = VarianceThreshold(threshold=0.01)
    X_var = selector.fit_transform(X_scaled)
    selected_columns = X.columns[selector.get_support()]
    X_var = pd.DataFrame(X_var, columns=selected_columns)
    joblib.dump(selector, "models/variance_filter.pkl")
    joblib.dump(list(selected_columns), "models/selected_columns.pkl")
    print(f"Variance filter sonrasi: {len(selected_columns)} kolon")

    # ── KLİNİK VERİ ────────────────────────────────────────
    meta = pd.read_csv(CLINICAL_PATH)
    X_var = X_var.copy()
    X_var['ID'] = patient_ids.str.replace('_nifti', '').str.replace('UCSF-PDGM-0', 'UCSF-PDGM-')
    X_var['Patient_ID'] = patient_ids.values

    meta['Sex_enc'] = (meta['Sex'] == 'M').astype(int)
    meta['IDH_enc'] = (meta['IDH'] == 'wildtype').astype(int)
    mgmt_map = {'positive': 1, 'negative': 0, 'indeterminate': -1}
    meta['MGMT_enc'] = meta['MGMT'].map(mgmt_map)

    klinik = meta[['ID', 'Age At MRI', 'Sex_enc', 'IDH_enc', 'MGMT_enc',
                   'Grade', 'Survival Days', '1-dead 0-alive']]

    merged = pd.merge(X_var, klinik, on='ID', how='inner')
    merged = merged.dropna(subset=['Grade'])
    print(f"Klinik veriyle birlesen hasta sayisi: {len(merged)}")

    hedef_cols = ['Grade', 'Survival Days', '1-dead 0-alive']
    klinik_cols = ['Age At MRI', 'Sex_enc', 'IDH_enc', 'MGMT_enc']
    rad_cols = [c for c in selected_columns if c in merged.columns]

    X_rad = merged[rad_cols]
    X_klinik = merged[klinik_cols]
    X_birlesik = pd.concat([X_rad, X_klinik], axis=1)
    y = merged['Grade'].values

    print(f"\nRFECV icin: {len(rad_cols)} radiomics + {len(klinik_cols)} klinik")
    print("Hedef: Grade")
    print("Kac ozellik secilecegi otomatik belirleniyor (RFECV)...")

    # RFECV — kac ozellik secilecegini cross-validation ile otomatik buluyor
    rfecv = RFECV(
        estimator=RandomForestClassifier(n_estimators=100, random_state=42),
        step=50,
        cv=StratifiedKFold(n_splits=5),
        scoring='accuracy',
        min_features_to_select=10
    )
    rfecv.fit(X_birlesik, y)

    # Sadece radiomics kolonlarindan secilenleri al
    selected_all = X_birlesik.columns[rfecv.support_].tolist()
    final_features = [f for f in selected_all if f in rad_cols]

    print(f"RFECV optimum ozellik sayisi: {rfecv.n_features_}")
    print(f"Bunun radiomics kismi: {len(final_features)} ozellik")

    joblib.dump(final_features, "models/final_columns.pkl")

    # ── KAYDET ─────────────────────────────────────────────
    df_final = merged[['Patient_ID'] + final_features + hedef_cols].copy()
    df_final.to_csv(OUTPUT_PATH, index=False)

    print(f"\nTamamlandi: {len(df_final)} hasta")
    print(f"  Radiomics ozellik: {len(final_features)}")
    print(f"  Hedef kolon: Grade, Survival Days, 1-dead 0-alive")
    print(f"Kaydedilen: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()