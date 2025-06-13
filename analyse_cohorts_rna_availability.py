import pandas as pd
import os

# --- Configuration ---
# 1. Replace 'your_excel_file.xlsx' with the exact name of your Excel file.
excel_file_path = "C:/Users/Mahon/Documents/Research/Unorgansied/TCGA_Patient_List_annotated.xlsx"

# 2. If your data is not on the first sheet, replace 'Sheet1' with the correct sheet name.
sheet_name = 'Sheet1'
# -------------------


def analyze_rna_availability(file_path, sheet):
    """
    Reads an Excel file, groups by cohort, and calculates the percentage
    of RNA-seq data availability for each cohort.
    """
    # Check if the file exists before trying to read it
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        print("Please make sure the script is in the same directory as your Excel file, or provide the full path.")
        return

    try:
        # --- MODIFIED LINE ---
        # Added 'header=1' to specify that the column names are in the second row.
        df = pd.read_excel(file_path, sheet_name=sheet, header=1)

        # Ensure column names are clean (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()

        # Check if the required columns exist
        if 'COHORT' not in df.columns or 'RNAseq' not in df.columns:
            print(f"Error: The Excel sheet must contain 'COHORT' and 'RNAseq' columns.")
            print(f"Found columns: {df.columns.tolist()}")
            return

        print("Successfully loaded the Excel file. Analyzing data...")

        # Normalize the 'RNAseq' column for accurate counting (e.g., converts 'Yes' to 'yes')
        df['RNAseq_normalized'] = df['RNAseq'].str.lower().str.strip()

        # Group by the 'COHORT' column and calculate the percentage
        # This works by counting the 'yes' values and dividing by the total size of the group.
        rna_summary = df.groupby('COHORT')['RNAseq_normalized'].apply(
            lambda x: (x == 'yes').sum() / len(x) * 100
        ).reset_index(name='RNAseq_Available_Percent')

        # Sort the results for better readability
        rna_summary = rna_summary.sort_values(by='RNAseq_Available_Percent', ascending=False)

        print("\n--- Analysis Complete ---")
        print("Percentage of patients with RNA-seq data available for each cohort:")
        # Print the resulting table in a clean format
        print(rna_summary.to_string(index=False))

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")

if __name__ == '__main__':
    analyze_rna_availability(excel_file_path, sheet_name)