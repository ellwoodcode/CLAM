import os
import pandas as pd

EVAL_DIR = './eval_results'
OUT_FILE = 'all_summaries.csv'

all_rows = []

for root, dirs, files in os.walk(EVAL_DIR):
    for file in files:
        if file == 'summary.csv':
            full_path = os.path.join(root, file)
            try:
                df = pd.read_csv(full_path)
                # Parse model and cohort from path, e.g. EVAL_IMS1_on_Georgia
                parts = os.path.basename(root).split('_on_')
                if len(parts) == 2:
                    model = parts[0].replace('EVAL_', '')
                    cohort = parts[1]
                else:
                    model, cohort = "UNKNOWN", "UNKNOWN"
                df['model'] = model
                df['cohort'] = cohort
                all_rows.append(df)
                print(f"✅ Loaded {full_path}")
            except Exception as e:
                print(f"❌ Failed to read {full_path}: {e}")

if all_rows:
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(OUT_FILE, index=False)
    print(f"\n✅ Compiled summary saved to: {OUT_FILE}")
else:
    print("\n⚠️ No summary.csv files found.")
