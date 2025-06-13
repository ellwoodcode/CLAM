import os
import pandas as pd

# Set the folder containing your CSVs
csv_dir = "C:/Users/Mahon/Documents/Research/CLAM/Labels/Textual/"  
results_dir = "C:/Users/Mahon/Documents/Research/CLAM/Labels/Tangle/"  # UPDATE if needed

# Map specific filenames to cohort names if not directly derivable from filename
special_cases = {
    'labels_ims1_filtered.csv': 'IMS1',
    'labels_ims2_filtered.csv': 'IMS2',
}

def infer_cohort(filename):
    """Infer cohort name from filename, or use special_cases mapping."""
    if filename in special_cases:
        return special_cases[filename]
    name = filename.replace('_filtered.csv', '').replace('.csv', '')
    # Remove "labels_" prefix if present
    if name.startswith('labels_'):
        name = name[len('labels_'):]
    return name

for fname in os.listdir(csv_dir):
    if fname.endswith('.csv') and not fname.endswith('_cohort.csv'):
        cohort = infer_cohort(fname)
        in_path = os.path.join(csv_dir, fname)
        out_path = os.path.join(results_dir, fname.replace('.csv', '_cohort.csv'))
        print(f"Processing {fname} as cohort '{cohort}'")
        try:
            df = pd.read_csv(in_path)
            df['cohort'] = cohort
            df.to_csv(out_path, index=False)
            print(f"-> Saved to {out_path}")
        except Exception as e:
            print(f"Failed to process {fname}: {e}")
