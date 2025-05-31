import openslide
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def view_wsi(wsi_path, view_level=None, downsample_factor=32.0):
    """
    Opens and displays a Whole Slide Image (WSI) at a specified level or downsample factor.

    Args:
        wsi_path (str): Path to the WSI file.
        view_level (int, optional): The pyramid level to display. 
                                   If None, downsample_factor is used. Defaults to None.
        downsample_factor (float, optional): Factor to downsample the image if view_level is not specified.
                                          Ignored if view_level is provided. Defaults to 32.0.
    """
    if not os.path.exists(wsi_path):
        print(f"Error: WSI file not found at {wsi_path}")
        return

    try:
        wsi = openslide.OpenSlide(wsi_path)
        print(f"Successfully opened: {os.path.basename(wsi_path)}")
        print(f"Vendor: {wsi.properties.get(openslide.PROPERTY_NAME_VENDOR, 'N/A')}")
        print(f"Dimensions at level 0: {wsi.level_dimensions[0]}")
        print(f"Number of levels: {wsi.level_count}")
        print(f"Level downsamples: {wsi.level_downsamples}")

        if view_level is not None:
            if view_level < 0 or view_level >= wsi.level_count:
                print(f"Error: Invalid view_level {view_level}. Available levels: 0 to {wsi.level_count - 1}")
                wsi.close()
                return
            level_to_read = view_level
            dims = wsi.level_dimensions[level_to_read]
            print(f"Reading level {level_to_read} with dimensions {dims} (Downsample: {wsi.level_downsamples[level_to_read]})")
        else:
            level_to_read = wsi.get_best_level_for_downsample(downsample_factor)
            dims = wsi.level_dimensions[level_to_read]
            print(f"Target downsample factor: {downsample_factor}")
            print(f"Reading best level {level_to_read} with dimensions {dims} (Actual downsample: {wsi.level_downsamples[level_to_read]})")
        
        # Read the entire region for the chosen level
        # This might still be large for very high-resolution levels on huge slides
        # For extremely large slides at low downsample, consider reading a specific region
        img_pil = wsi.read_region((0,0), level_to_read, dims)
        
        # Convert PIL image to NumPy array for Matplotlib
        img_np = np.array(img_pil.convert("RGB")) # Ensure it's RGB

        wsi.close() # Close the WSI object

        # Display the image
        plt.figure(figsize=(12, 10)) # Adjust figure size as needed
        plt.imshow(img_np)
        plt.title(f"{os.path.basename(wsi_path)}\nLevel: {level_to_read}, Dimensions: {dims}")
        plt.axis('off') # Turn off axis numbers and ticks
        plt.tight_layout()
        plt.show()

    except openslide.OpenSlideError as e:
        print(f"OpenSlide Error: Could not open or read file {wsi_path}. Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display a Whole Slide Image (SVS file).")
    parser.add_argument("wsi_path", type=str, help="Full path to the WSI file (e.g., .svs).")
    parser.add_argument("--level", type=int, default=None, 
                        help="Specific pyramid level to display (e.g., 0, 1, 2 ...). Overrides --downsample.")
    parser.add_argument("--downsample", type=float, default=32.0, 
                        help="Approximate downsample factor to use if --level is not specified (e.g., 32 for 32x downsampling).")
    
    args = parser.parse_args()
    
    view_wsi(args.wsi_path, view_level=args.level, downsample_factor=args.downsample)