import os
import pandas as pd
from deepface import DeepFace
from tqdm import tqdm

# Path to your dataset images folder
DATASET_PATH = "part1\dataset"
OUTPUT_CSV = "labels.csv"

def generate_labels():
    data = []

    for img_name in tqdm(os.listdir(DATASET_PATH)):
        if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(DATASET_PATH, img_name)

            try:
                analysis = DeepFace.analyze(
                    img_path=img_path,
                    actions=["age", "gender", "race"],
                    enforce_detection=False
                )

                age = analysis[0]["age"]
                gender = analysis[0]["dominant_gender"]
                skin_tone = analysis[0]["dominant_race"]

                data.append([img_name, age, gender, skin_tone])

            except Exception as e:
                print(f"[ERROR] Could not analyze {img_name}: {e}")

    # Save results to CSV
    df = pd.DataFrame(data, columns=["filename", "age", "gender", "skin_tone"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Labels saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    generate_labels()
