"""
Convert CSV paths from local/WSL format to Kaggle format.

This script fixes image paths in the CSV to match Kaggle's directory structure.

Usage:
    python scripts/kaggle/convert_csv_paths.py \
        --input /path/to/original.csv \
        --output /path/to/kaggle_ready.csv
"""

import pandas as pd
import argparse
import os


def convert_paths_to_kaggle(input_csv: str, output_csv: str):
    """
    Convert local paths to Kaggle paths in CSV.
    
    Args:
        input_csv: Path to original CSV with local paths
        output_csv: Path to save converted CSV
    """
    print(f"Loading CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Expected column names (adjust if yours are different)
    rgb_col = 'rgb_frame_path' if 'rgb_frame_path' in df.columns else 'rgb_frame'
    ir_col = 'ir_frame_path' if 'ir_frame_path' in df.columns else 'ir_frame'
    
    if rgb_col not in df.columns or ir_col not in df.columns:
        print(f"Error: Expected columns '{rgb_col}' and '{ir_col}' not found!")
        print(f"Available columns: {list(df.columns)}")
        return False
    
    # Show example of original paths
    print("\n" + "="*70)
    print("BEFORE CONVERSION")
    print("="*70)
    print(f"RGB example: {df[rgb_col].iloc[0]}")
    print(f"IR example:  {df[ir_col].iloc[0]}")
    
    # Define path patterns to replace
    patterns_to_replace = [
        # WSL paths
        '/mnt/c/Users/T2430451/data/datasets/adaptive-median-gpu',
        '/mnt/c/Users/T2430451/data/adaptive-median-gpu',
        
        # Windows paths  
        'C:/Users/T2430451/data/datasets/adaptive-median-gpu',
        'C:/Users/T2430451/data/adaptive-median-gpu',
        'C:\\Users\\T2430451\\data\\datasets\\adaptive-median-gpu',
        'C:\\Users\\T2430451\\data\\adaptive-median-gpu',
        
        # Generic patterns
        '/mnt/c/Users/T2430451/data',
        'C:/Users/T2430451/data',
        'C:\\Users\\T2430451\\data',
    ]
    
    # Kaggle target path (note: nested adaptive-median-gpu directory)
    kaggle_base = '/kaggle/input/flame2/adaptive-median-gpu/adaptive-median-gpu'
    
    # Convert paths
    print("\nConverting paths...")
    for pattern in patterns_to_replace:
        df[rgb_col] = df[rgb_col].str.replace(pattern, kaggle_base, regex=False)
        df[ir_col] = df[ir_col].str.replace(pattern, kaggle_base, regex=False)
    
    # Also handle case where paths might be relative or just filenames with subdirs
    # Pattern: "254p RGB Images/254p RGB Frame (123).jpg"
    # Should become: "/kaggle/input/flame2/adaptive-median-gpu/254p RGB Images/254p RGB Frame (123).jpg"
    
    # If path doesn't start with /, prepend kaggle base
    mask = ~df[rgb_col].str.startswith('/')
    if mask.any():
        df.loc[mask, rgb_col] = kaggle_base + '/' + df.loc[mask, rgb_col]
    
    mask = ~df[ir_col].str.startswith('/')
    if mask.any():
        df.loc[mask, ir_col] = kaggle_base + '/' + df.loc[mask, ir_col]
    
    # Show example of converted paths
    print("\n" + "="*70)
    print("AFTER CONVERSION")
    print("="*70)
    print(f"RGB example: {df[rgb_col].iloc[0]}")
    print(f"IR example:  {df[ir_col].iloc[0]}")
    
    # Verify all paths now point to Kaggle
    rgb_ok = df[rgb_col].str.startswith('/kaggle/input/flame2/').all()
    ir_ok = df[ir_col].str.startswith('/kaggle/input/flame2/').all()
    
    if not rgb_ok or not ir_ok:
        print("\n⚠ Warning: Some paths may not have been converted correctly!")
        if not rgb_ok:
            print(f"  RGB paths not starting with /kaggle/input/flame2/:")
            print(df[~df[rgb_col].str.startswith('/kaggle/input/flame2/')][rgb_col].head())
        if not ir_ok:
            print(f"  IR paths not starting with /kaggle/input/flame2/:")
            print(df[~df[ir_col].str.startswith('/kaggle/input/flame2/')][ir_col].head())
    else:
        print("\n✓ All paths successfully converted to Kaggle format!")
    
    # Save converted CSV
    print(f"\nSaving converted CSV: {output_csv}")
    df.to_csv(output_csv, index=False)
    
    print(f"✓ Done! Converted {len(df)} rows.")
    print(f"\nNext steps:")
    print(f"  1. Upload {output_csv} to Kaggle as part of your dataset")
    print(f"  2. Update scripts/kaggle/config.py if needed:")
    print(f"     CSV_PATH = '/kaggle/input/your-dataset/{os.path.basename(output_csv)}'")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV paths to Kaggle format")
    parser.add_argument('--input', required=True, help='Input CSV with local paths')
    parser.add_argument('--output', required=True, help='Output CSV with Kaggle paths')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        exit(1)
    
    success = convert_paths_to_kaggle(args.input, args.output)
    exit(0 if success else 1)
