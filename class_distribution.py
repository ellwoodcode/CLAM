import pandas as pd
import os

# === SET THIS TO YOUR CSV ROOT DIRECTORY ===
csv_root_dir = "C:/Users/Mahon/Documents/Research/CLAM/Labels/Boolean"

# === OUTPUT FILE ===
summary_csv = os.path.join(csv_root_dir, "class_distribution_summary.csv")

# === STORAGE ===
summary_data = []

# === ITERATE OVER CSV FILES ===
for filename in os.listdir(csv_root_dir):
    if filename.endswith(".csv"):
        filepath = os.path.join(csv_root_dir, filename)
        df = pd.read_csv(filepath)

        if not {'case_id', 'slide_id', 'label'}.issubset(df.columns):
            print(f"Skipping {filename}: required columns missing.")
            continue

        # Slide-level
        slide_counts = df['label'].value_counts().to_dict()
        total_slides = sum(slide_counts.values())
        normal_slides = slide_counts.get(0, 0)
        tumor_slides = slide_counts.get(1, 0)
        normal_slide_pct = (normal_slides / total_slides) * 100 if total_slides else 0
        tumor_slide_pct = (tumor_slides / total_slides) * 100 if total_slides else 0

        # Case-level
        case_labels = df.groupby('case_id')['label'].max()
        case_counts = case_labels.value_counts().to_dict()
        total_cases = sum(case_counts.values())
        normal_cases = case_counts.get(0, 0)
        tumor_cases = case_counts.get(1, 0)
        normal_case_pct = (normal_cases / total_cases) * 100 if total_cases else 0
        tumor_case_pct = (tumor_cases / total_cases) * 100 if total_cases else 0

        # Append row
        summary_data.append({
            'file': filename,
            'normal_slides': normal_slides,
            'tumor_slides': tumor_slides,
            'normal_slide_pct': f"{normal_slide_pct:.2f}%",
            'tumor_slide_pct': f"{tumor_slide_pct:.2f}%",
            'normal_cases': normal_cases,
            'tumor_cases': tumor_cases,
            'normal_case_pct': f"{normal_case_pct:.2f}%",
            'tumor_case_pct': f"{tumor_case_pct:.2f}%"
        })

# === SAVE SUMMARY ===
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(summary_csv, index=False)

print("Summary written to:")
print(summary_csv)