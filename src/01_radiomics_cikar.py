import os
import pandas as pd
import numpy as np
from radiomics import featureextractor
from tqdm import tqdm
import SimpleITK as sitk

CONFIG_PATH = "config/radiomics_config.yaml"
DATASET_PATH = "data/raw_dataset"
OUTPUT_PATH = "data/processed/radiomics.csv"

def extract_patient_features(patient_folder):
    try:
        extractor = featureextractor.RadiomicsFeatureExtractor(CONFIG_PATH)
    except Exception as e:
        print(f"YAML Hatasi: {e}")
        return None

    patient_id = os.path.basename(patient_folder)
    features = {"Patient_ID": patient_id}
    files = os.listdir(patient_folder)

    mask_file = next((f for f in files if "tumor_segmentation" in f.lower()), None)
    if mask_file is None:
        mask_file = next((f for f in files if "brain_segmentation" in f.lower()), None)

    if mask_file is None:
        print(f"Mask bulunamadi: {patient_id}")
        return None

    print(f"\n{patient_id} - Mask: {mask_file}")
    mask_path = os.path.join(patient_folder, mask_file)

    mask_sitk = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask_sitk)
    unique_values = np.unique(mask_array)
    print(f"   Mask etiketleri: {unique_values}")

    if np.sum(mask_array) == 0:
        print(f"Mask tamamen bos: {patient_id}")
        return None

    # Label'i otomatik bul (0 haric ilk deger)
    unique_labels = [v for v in unique_values if v != 0]
    extractor.settings['label'] = int(unique_labels[0])
    print(f"   Kullanilan label: {int(unique_labels[0])}")

    feature_count = 0
    for file in files:
        if not file.endswith(".nii.gz") or file == mask_file:
            continue

        image_path = os.path.join(patient_folder, file)
        sequence_name = file.split("_")[-1].replace(".nii.gz", "")

        try:
            img_sitk = sitk.ReadImage(image_path)

            if img_sitk.GetSize() != mask_sitk.GetSize():
                print(f"   Boyut uyumsuz - {sequence_name}: img{img_sitk.GetSize()} mask{mask_sitk.GetSize()}")
                continue

            print(f"   {sequence_name} isleniyor...")

            result = extractor.execute(image_path, mask_path)

            for key, value in result.items():
                if "diagnostics" not in key:
                    features[f"{sequence_name}_{key}"] = value
                    feature_count += 1

        except Exception as e:
            print(f"   Hata ({sequence_name}): {e}")

    print(f"   Toplam {feature_count} feature cikarildi.")

    if feature_count == 0:
        print(f"Hic feature cikarilamadi: {patient_id}")
        return None

    return features

def main():
    if not os.path.exists(os.path.dirname(OUTPUT_PATH)):
        os.makedirs(os.path.dirname(OUTPUT_PATH))

    all_features = []
    patients = [p for p in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, p))]

    print(f"{len(patients)} hasta icin islem basliyor...")

    for patient in tqdm(patients):
        patient_path = os.path.join(DATASET_PATH, patient)
        feat = extract_patient_features(patient_path)
        if feat:
            all_features.append(feat)

    if all_features:
        df = pd.DataFrame(all_features)
        df = df.reindex(sorted(df.columns), axis=1)
        cols = ['Patient_ID'] + [c for c in df.columns if c != 'Patient_ID']
        df = df[cols]
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nTamamlandi. Kaydedilen: {OUTPUT_PATH}")
        print(f"{len(df)} hasta, {len(df.columns)-1} feature")
    else:
        print("\nIslenecek gecerli veri bulunamadi.")

if __name__ == "__main__":
    main()