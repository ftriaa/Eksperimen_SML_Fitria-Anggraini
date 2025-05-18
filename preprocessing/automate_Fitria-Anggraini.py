import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(raw_data_path, output_csv_path, artifacts_path):
    # Load dataset
    df = pd.read_csv(raw_data_path)

    # Drop kolom 'id'
    df.drop('id', axis=1, inplace=True)

    # Drop baris gender == 'Other'
    df = df[df['gender'] != 'Other']

    # Gabungkan kategori minoritas di kolom 'work_type'
    df['work_type'] = df['work_type'].replace({
        'children': 'Other',
        'Never_worked': 'Other'
    })

    # Imputasi missing value kolom 'bmi' dengan median
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())

    # Label Encoding untuk kolom tertentu
    label_encoders = {}
    label_enc_cols = ['ever_married', 'Residence_type', 'gender']
    for col in label_enc_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # simpan encoder per kolom

    # One-Hot Encoding untuk kolom kategorikal lainnya
    df = pd.get_dummies(df, columns=['work_type', 'smoking_status'], drop_first=True)

    # Scaling fitur numerik
    num_cols = ['age', 'avg_glucose_level', 'bmi']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Pastikan folder output CSV ada
    output_dir = os.path.dirname(output_csv_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Simpan dataset hasil preprocessing
    df.to_csv(output_csv_path, index=False)
    print(f"Preprocessing selesai. Data disimpan di: {output_csv_path}")

    # Pastikan folder artefak ada
    artifact_dir = os.path.dirname(artifacts_path)
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    # Simpan artefak scaler dan label encoder
    artefak = {
        'scaler': scaler,
        'label_encoders': label_encoders
    }
    joblib.dump(artefak, artifacts_path)
    print(f"Artefak preprocessing disimpan di: {artifacts_path}")

if __name__ == "__main__":
    RAW_DATA_PATH = "../healthcare-dataset-stroke-data.csv"
    OUTPUT_CSV = "stroke_dataset_preprocessing.csv"
    ARTIFACTS_PATH = "joblib/preprocessing_artifacts.joblib"

    preprocess_data(RAW_DATA_PATH, OUTPUT_CSV, ARTIFACTS_PATH)
