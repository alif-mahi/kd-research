import pandas as pd
import cv2
import numpy as np
import os
from tqdm import tqdm

def contrast_stretching(img):
    """
    Performs contrast stretching (normalization) on an image.
    """
    # Normalize to 0-255
    norm_img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return norm_img

def process_images(input_csv, output_csv, output_base_dir):
    if not os.path.exists(input_csv):
        print(f"Error: Input file '{input_csv}' not found.")
        return

    df = pd.read_csv(input_csv)
    
    # Prepare output lists
    new_rgb_paths = []
    new_ir_paths = []
    
    # Ensure output directories exist
    rgb_out_dir = os.path.join(output_base_dir, '254p RGB Images')
    ir_out_dir = os.path.join(output_base_dir, '254p Thermal Images')
    os.makedirs(rgb_out_dir, exist_ok=True)
    os.makedirs(ir_out_dir, exist_ok=True)
    
    print(f"Processing {len(df)} images...")
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # Resolve paths relative to the CSV location
        csv_dir = os.path.dirname(input_csv)
        rgb_path = os.path.normpath(os.path.join(csv_dir, row['rgb_frame']))
        ir_path = os.path.normpath(os.path.join(csv_dir, row['ir_frame']))
        
        # Read images
        rgb_img = cv2.imread(rgb_path)
        ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE) # Thermal is usually grayscale
        
        if rgb_img is None:
            print(f"Warning: Could not read RGB image at {rgb_path}")
            new_rgb_paths.append(None)
        else:
            # Process RGB
            # Split channels, stretch each, merge
            b, g, r = cv2.split(rgb_img)
            b_stretched = contrast_stretching(b)
            g_stretched = contrast_stretching(g)
            r_stretched = contrast_stretching(r)
            rgb_stretched = cv2.merge((b_stretched, g_stretched, r_stretched))
            
            # Save
            rgb_filename = os.path.basename(rgb_path)
            save_path_rgb = os.path.join(rgb_out_dir, rgb_filename)
            cv2.imwrite(save_path_rgb, rgb_stretched)
            
            # Store relative path for CSV (relative to dataframes folder)
            # output_base_dir is datasets/contrast-stretched
            # We want ../datasets/contrast-stretched/...
            rel_path_rgb = os.path.join('..', output_base_dir, '254p RGB Images', rgb_filename)
            new_rgb_paths.append(rel_path_rgb)

        if ir_img is None:
            print(f"Warning: Could not read IR image at {ir_path}")
            new_ir_paths.append(None)
        else:
            # Process IR
            ir_stretched = contrast_stretching(ir_img)
            
            # Save
            ir_filename = os.path.basename(ir_path)
            save_path_ir = os.path.join(ir_out_dir, ir_filename)
            cv2.imwrite(save_path_ir, ir_stretched)
            
            rel_path_ir = os.path.join('..', output_base_dir, '254p Thermal Images', ir_filename)
            new_ir_paths.append(rel_path_ir)

    # Update DataFrame
    df['rgb_frame'] = new_rgb_paths
    df['ir_frame'] = new_ir_paths
    
    # Remove rows with failures
    df = df.dropna(subset=['rgb_frame', 'ir_frame'])
    
    # Save new CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved processed images to '{output_base_dir}' and CSV to '{output_csv}'.")

if __name__ == "__main__":
    input_csv = 'dataframes/prototype.csv'
    output_csv = 'dataframes/contrast_stretched.csv'
    output_base_dir = 'datasets/contrast-stretched'
    
    process_images(input_csv, output_csv, output_base_dir)
