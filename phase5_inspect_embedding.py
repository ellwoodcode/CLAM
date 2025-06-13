import numpy as np
import argparse
import os

def inspect_embedding_file(file_path):
    """
    Loads a .npy file and prints its properties.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    try:
        # Load the data from the .npy file
        embedding = np.load(file_path)

        print(f"\n--- Inspection Report for: {os.path.basename(file_path)} ---")
        
        # --- 1. Print Properties ---
        print(f"Data Type: {embedding.dtype}")
        print(f"Shape: {embedding.shape}")
        
        # --- 2. Print Content Summary ---
        if embedding.size > 0:
            print(f"Min Value: {np.min(embedding):.6f}")
            print(f"Max Value: {np.max(embedding):.6f}")
            print(f"Mean Value: {np.mean(embedding):.6f}")
            
            # Print the first few values to get a sense of the data
            num_values_to_show = min(10, embedding.size)
            print(f"First {num_values_to_show} values: {embedding[:num_values_to_show]}")
        else:
            print("The array is empty.")
            
        print("--- End of Report ---\n")

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect a .npy embedding file.')
    parser.add_argument('--path', type=str, required=True,
                        help='Full path to the .npy file you want to inspect.')
    
    args = parser.parse_args()
    inspect_embedding_file(args.path)