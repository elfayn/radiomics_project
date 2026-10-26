import os
import shutil
import warnings
import pandas as pd
import numpy as np
import joblib
import SimpleITK as sitk
from radiomics import featureextractor

warnings.filterwarnings("ignore")

CONFIG_PATH = "config/radiomics_config.yaml"
NEW_PATIENT_PATH = "data/new_patients"
MAIN_DATASET_PATH = "data/raw_dataset"
OUTPUT_PATH = "data/processed/temizlenmis.csv"

def extract_features(patient_folder):
    extractor = featureextractor.RadiomicsFeatureExtractor(CONFIG_PATH)

    files = os.listdir(patient_folder)

    mask_file = next((f for f in files if "tumor_segmentation" in f.lower()), None)
    if mask_file is None:
        mask_file = next((f for f in files if "brain_segmentation" in f.lower()), None)

    if mask_file is None:
        print("Mask bulunamadi.")
        return None

    mask_path = os.path.join(patient_folder, mask_file)

    mask_sitk = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask_sitk)
    unique_labels = [v for v in np.unique(mask_array) if v != 0]

    if len(unique_labels) == 0:
        print("Mask bos, atlaniyor.")
        return None

    extractor.settings['label'] = int(unique_labels[0])

    features = {}

    for file in files:
        if not file.endswith(".nii.gz") or file == mask_file:
            continue

        if "segmentation" in file.lower():
            continue

        image_path = os.path.join(patient_folder, file)
        sequence = file.split("_")[-1].replace(".nii.gz", "")

        try:
            img_sitk = sitk.ReadImage(image_path)
            if img_sitk.GetSize() != mask_sitk.GetSize():
                print(f"   Boyut uyumsuz, atlaniyor: {sequence}")
                continue

            print(f"   {sequence} isleniyor...")
            result = extractor.execute(image_path, mask_path)

            for key, value in result.items():
                if "diagnostics" not in key:
                    features[f"{sequence}_{key}"] = value

        except Exception as e:
            print(f"   Hata ({sequence}): {e}")

    return features


def main():

    scaler = joblib.load("models/scaler.pkl")
    variance_selector = joblib.load("models/variance_filter.pkl")
    scaler_columns = joblib.load("models/selected_columns.pkl")
    final_columns = joblib.load("models/final_columns.pkl")

    scaler_columns_arr = np.array(scaler_columns)

    new_rows = []
    processed_patients = []

    for patient in os.listdir(NEW_PATIENT_PATH):
        patient_path = os.path.join(NEW_PATIENT_PATH, patient)

        if not os.path.isdir(patient_path):
            continue

        print(f"\nIsleniyor: {patient}")

        feat = extract_features(patient_path)
        if feat is None:
            continue

        df = pd.DataFrame([feat])

        # Kompleks sayi temizligi
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(lambda v: float(complex(v).real)
                                        if isinstance(v, str) else v)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Eksik kolonlari doldur
        for col in scaler_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[scaler_columns]

        # Normalize ve filtrele
        X_scaled = scaler.transform(df)
        X_var = variance_selector.transform(X_scaled)

        df_var = pd.DataFrame(X_var, columns=scaler_columns_arr[variance_selector.get_support()])

        # Sadece final kolonlari al
        for col in final_columns:
            if col not in df_var.columns:
                df_var[col] = 0
        df_final = df_var[final_columns].copy()

        df_final.insert(0, "Patient_ID", patient)

        # Hedef kolonlar yeni hasta icin bos — klinik veri sonradan eklenecek
        df_final['Grade'] = np.nan
        df_final['Survival Days'] = np.nan
        df_final['1-dead 0-alive'] = np.nan

        new_rows.append(df_final)
        processed_patients.append((patient, patient_path))

        print(f"Hesaplandi: {patient}")

    if new_rows:
        new_df = pd.concat(new_rows, ignore_index=True)
        old_df = pd.read_csv(OUTPUT_PATH)

        # Kolon sirasi ayni olmali
        new_df = new_df[old_df.columns]

        final_df = pd.concat([old_df, new_df], ignore_index=True)

        try:
            final_df.to_csv(OUTPUT_PATH, index=False)
            print(f"\nCSV kaydedildi: {len(new_rows)} yeni hasta eklendi.")
            print("Not: Grade, Survival Days, 1-dead 0-alive klinik veri bekleniyor.")

            for patient, patient_path in processed_patients:
                shutil.move(patient_path, os.path.join(MAIN_DATASET_PATH, patient))
                print(f"Tasindi: {patient}")

        except PermissionError:
            print("\nHATA: temizlenmis.csv dosyasi acik! Lutfen Excel'i kapatin ve tekrar calistirin.")
            print("Hastalar tasinmadi, new_patients klasorunde bekliyor.")

    else:
        print("Yeni hasta bulunamadi.")


if __name__ == "__main__":
    main()