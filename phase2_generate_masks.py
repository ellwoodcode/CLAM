import torch
import torchvision.transforms.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import os
import glob
from tqdm import tqdm
import argparse
import sys
import h5py
import openslide

# Optional: for morphological operations if you want to add them
# import cv2

# --- Configuration via Arguments ---
def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Generate Pseudo-Ground Truth Masks On-The-Fly from WSIs and H5 Coords.")
    parser.add_argument('--source_wsi_dir', type=str, required=True,
                        help='Base directory containing original WSI files (e.g., SVS).')
    parser.add_argument('--input_h5_coord_dir', type=str, required=True,
                        help='Base directory containing H5 files with patch coordinates (output of Phase 1).')
    parser.add_argument('--output_mask_dir', type=str, required=True,
                        help='Base directory to save generated binary mask PNGs.')
    parser.add_argument('--wsi_file_extension', type=str, default=".svs",
                        help='File extension for your WSI files (e.g., .svs, .tif).')
    parser.add_argument('--model_name', type=str, default="facebook/dinov2-with-registers-large",
                        help='Name of the pre-trained DINOv2/TRIDENT model from Hugging Face or local path.')
    parser.add_argument('--patch_size_to_extract', type=int, default=256,
                        help='The patch size (height and width) to read from the WSI.')
    parser.add_argument('--patch_level_to_extract', type=int, default=0,
                        help='The WSI pyramid level from which to extract patches.')
    parser.add_argument('--attention_threshold', type=float, default=0.6,
                        help='Threshold (0.0-1.0) for binarizing the normalized attention map.')
    parser.add_argument('--batch_size', type=int, default=32, # Batching PIL images before processor
                        help='Batch size for processing patches (number of PIL images to process together).')
    parser.add_argument('--num_workers', type=int, default=0, # Usually 0 for on-the-fly WSI reading unless dataset is complex
                        help='Number of workers for DataLoader (mostly for file listing here).')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disable CUDA even if available.')
    parser.add_argument('--auto_skip_masks', action='store_true', default=False,
                        help='Skip mask generation if the mask file already exists.')
    return parser.parse_args()

# --- Main Script Logic ---
def main(args):
    print("--- Phase 2: Pseudo-Ground Truth Mask Generation (On-The-Fly from WSI) ---")
    print(f"Source WSI Directory: {args.source_wsi_dir}")
    print(f"Input H5 Coordinate Directory: {args.input_h5_coord_dir}")
    print(f"Output Mask Directory: {args.output_mask_dir}")
    print(f"Model: {args.model_name}")
    print(f"Patch Size to Extract: {args.patch_size_to_extract}x{args.patch_size_to_extract}")
    print(f"Patch Level to Extract: {args.patch_level_to_extract}")
    print(f"Attention Threshold: {args.attention_threshold}")
    print(f"Processing Batch Size (PIL Images): {args.batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_mask_dir, exist_ok=True)

    print(f"Loading model: {args.model_name}...")
    try:
        processor = AutoImageProcessor.from_pretrained(args.model_name)
        model = AutoModel.from_pretrained(args.model_name)
        model.config.output_attentions = True
        model.to(device)
        model.eval()
        vit_patch_size = model.config.patch_size
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return

    h5_coord_files = sorted(glob.glob(os.path.join(args.input_h5_coord_dir, "*.h5")))
    if not h5_coord_files:
        print(f"No H5 coordinate files found in {args.input_h5_coord_dir}", file=sys.stderr)
        return

    for h5_coord_filepath in tqdm(h5_coord_files, desc="Processing Slides"):
        slide_id = os.path.splitext(os.path.basename(h5_coord_filepath))[0]
        wsi_file_path = os.path.join(args.source_wsi_dir, slide_id + args.wsi_file_extension)

        slide_mask_output_dir = os.path.join(args.output_mask_dir, slide_id)
        os.makedirs(slide_mask_output_dir, exist_ok=True)

        if not os.path.exists(wsi_file_path):
            print(f"WSI file {wsi_file_path} not found for slide {slide_id}. Skipping.", file=sys.stderr)
            continue

        try:
            wsi = openslide.OpenSlide(wsi_file_path)
            with h5py.File(h5_coord_filepath, 'r') as hf:
                if 'coords' not in hf:
                    print(f"'coords' dataset not found in {h5_coord_filepath}. Skipping slide {slide_id}.", file=sys.stderr)
                    wsi.close()
                    continue
                coordinates = hf['coords'][:]
        except Exception as e:
            print(f"Error opening WSI or H5 for slide {slide_id}: {e}", file=sys.stderr)
            if 'wsi' in locals() and wsi: wsi.close()
            continue
        
        print(f"Processing {len(coordinates)} patches for slide {slide_id}...")
        
        pil_image_batch = []
        coord_batch_info = []

        for patch_idx, coord in enumerate(tqdm(coordinates, desc=f"Slide {slide_id} Patches", leave=False)):
            x_coord, y_coord = int(coord[0]), int(coord[1])
            mask_filename = os.path.join(slide_mask_output_dir, f"mask_x{x_coord}_y{y_coord}_lvl{args.patch_level_to_extract}.png")

            if args.auto_skip_masks and os.path.exists(mask_filename):
                continue

            try:
                patch_pil = wsi.read_region((x_coord, y_coord), 
                                            args.patch_level_to_extract, 
                                            (args.patch_size_to_extract, args.patch_size_to_extract)).convert("RGB")
                pil_image_batch.append(patch_pil)
                coord_batch_info.append({'x': x_coord, 'y': y_coord, 'mask_path': mask_filename})
            except Exception as e_read:
                print(f"Error reading patch at ({x_coord},{y_coord}) for {slide_id}: {e_read}", file=sys.stderr)
                continue

            if len(pil_image_batch) == args.batch_size or patch_idx == len(coordinates) - 1:
                if not pil_image_batch: # If batch is empty (e.g. all skipped or error)
                    continue

                inputs = processor(images=pil_image_batch, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    attentions = outputs.attentions
                
                last_layer_attentions = attentions[-1]

                h_processed = inputs.pixel_values.shape[2]
                w_processed = inputs.pixel_values.shape[3]
                h_featmap = h_processed // vit_patch_size
                w_featmap = w_processed // vit_patch_size
                num_patch_tokens = h_featmap * w_featmap

                for i in range(last_layer_attentions.size(0)): # Iterate through images in the batch
                    current_coord_info = coord_batch_info[i]
                    
                    cls_attentions_to_patches = last_layer_attentions[i, :, 0, 1:num_patch_tokens+1]
                    avg_cls_attentions = cls_attentions_to_patches.mean(dim=0)

                    if avg_cls_attentions.shape[0] != num_patch_tokens:
                        print(f"Warning: Mismatch in patch tokens for patch at "
                              f"({current_coord_info['x']},{current_coord_info['y']}) in {slide_id}. Skipping.", file=sys.stderr)
                        continue
                    
                    try:
                        attention_map_2d = avg_cls_attentions.reshape(h_featmap, w_featmap)
                    except RuntimeError as e_reshape:
                        print(f"Error reshaping attention for patch at "
                              f"({current_coord_info['x']},{current_coord_info['y']}) in {slide_id}: {e_reshape}. Skipping.", file=sys.stderr)
                        continue
                    
                    resized_attention_map_tensor = F.resize(
                        attention_map_2d.unsqueeze(0).unsqueeze(0),
                        size=(args.patch_size_to_extract, args.patch_size_to_extract),
                        interpolation=T.InterpolationMode.BILINEAR,
                        antialias=True
                    ).squeeze()
                    
                    resized_attention_map_np = resized_attention_map_tensor.cpu().numpy()
                    map_min, map_max = np.min(resized_attention_map_np), np.max(resized_attention_map_np)
                    if map_max - map_min < 1e-6:
                        normalized_attention_map = np.zeros_like(resized_attention_map_np)
                    else:
                        normalized_attention_map = (resized_attention_map_np - map_min) / (map_max - map_min)
                    
                    binary_mask_np = (normalized_attention_map > args.attention_threshold).astype(np.uint8) * 255
                    binary_mask_pil = Image.fromarray(binary_mask_np, mode='L')
                    
                    try:
                        binary_mask_pil.save(current_coord_info['mask_path'])
                    except Exception as e_save:
                        print(f"Error saving mask {current_coord_info['mask_path']}: {e_save}", file=sys.stderr)
                
                pil_image_batch = [] # Reset batch
                coord_batch_info = [] # Reset batch info

        if 'wsi' in locals() and wsi:
             wsi.close()

    print("\n--- On-The-Fly Pseudo-Mask Generation Complete ---")
    print(f"Masks saved in subdirectories under: {args.output_mask_dir}")
    print(f"IMPORTANT: Please visually inspect a sample of the generated masks "
          f"and adjust --attention_threshold if necessary.")

if __name__ == '__main__':
    args = parse_args()
    main(args)