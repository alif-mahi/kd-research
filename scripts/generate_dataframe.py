import os
import pandas as pd
import re
import argparse
import sys

def build_frame_index(directory):
    """Create a dictionary mapping frame numbers to file paths."""
    frame_dict = {}
    if not os.path.exists(directory):
        print(f"Warning: Directory not found: {directory}")
        return frame_dict
        
    for filename in os.listdir(directory):
        # Extract frame number from filename like "254p RGB Frame (14586).jpg" or "254p Thermal Frame (1).jpg"
        match = re.search(r'\((\d+)\)\.jpg$', filename)
        if match:
            frame_num = int(match.group(1))
            frame_dict[frame_num] = os.path.join(directory, filename)
    return frame_dict

def parse_labels(label_file):
    """Convert label ranges to individual frame labels."""
    label_map = {}
    if not os.path.exists(label_file):
        print(f"Error: Label file not found: {label_file}")
        return label_map

    with open(label_file, 'r') as f:
        for line in f:
            # Skip empty lines or lines without tabs
            line = line.strip()
            if not line or '\t' not in line:
                continue
                
            parts = line.split('\t')
            # Skip lines that don't have exactly 3 parts
            if len(parts) != 3:
                continue
                
            start, end, flags = parts
            try:
                start = int(start)
                end = int(end)
                # Ensure flags has exactly 2 characters
                if len(flags) == 2:
                    fire = flags[0] == 'Y'
                    smoke = flags[1] == 'Y'
                    for frame_num in range(start, end + 1):
                        label_map[frame_num] = (fire, smoke)
            except ValueError:
                continue
    return label_map

def generate_dataframe(base_path, output_csv):
    """Generates the dataframe by matching RGB, Thermal images and Labels."""
    
    # Define paths relative to base_path
    rgb_path = os.path.join(base_path, "/mnt/c/Users/T2430451/data/datasets/original/254p RGB Images")
    ir_path = os.path.join(base_path, "/mnt/c/Users/T2430451/data/datasets/original/254p Thermal Images")
    label_file = os.path.join(base_path, "/mnt/c/Users/T2430451/data/dataframes/label.txt")
    
    # Step 1: Index frames
    print(f"Indexing RGB frames from {rgb_path}...")
    rgb_index = build_frame_index(rgb_path)
    print(f"Found {len(rgb_index)} RGB frames")

    print(f"Indexing Thermal frames from {ir_path}...")
    ir_index = build_frame_index(ir_path)
    print(f"Found {len(ir_index)} Thermal frames")

    # Step 2: Parse labels
    print(f"Parsing labels from {label_file}...")
    label_map = parse_labels(label_file)
    print(f"Processed labels for {len(label_map)} frames")

    # Step 3: Find common frames
    valid_frames = set(rgb_index.keys()) & set(ir_index.keys()) & set(label_map.keys())
    print(f"Found {len(valid_frames)} valid frames with both images and labels")

    if not valid_frames:
        print("Error: No valid frames found. Check your directories and label file.")
        return

    # Step 4: Build DataFrame
    data = []
    for frame_num in sorted(valid_frames):
        fire, smoke = label_map[frame_num]
        data.append({
            'id': frame_num,
            'rgb_frame': rgb_index[frame_num],
            'ir_frame': ir_index[frame_num],
            'fire': fire,
            'smoke': smoke
        })

    df = pd.DataFrame(data)

    # Step 5: Save to CSV
    if not df.empty:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"DataFrame successfully saved to {output_csv}")
        print(f"Total frames processed: {len(df)}")
    else:
        print("Warning: DataFrame is empty, no CSV saved.")

def main():
    parser = argparse.ArgumentParser(description="Generate dataset DataFrame from images and labels.")
    parser.add_argument("--base_path", default=".", help="Base path of the project (default: current directory)")
    parser.add_argument("--output", default="/mnt/c/Users/T2430451/data/dataframes/frame_labels.csv", help="Output CSV path (default: /mnt/c/Users/T2430451/data/dataframes/frame_labels.csv)")
    
    args = parser.parse_args()
    
    # Adjust base_path if we are running from scripts/ directory to match the original script's assumption of "../"
    # But better is to assume the user runs it from root or provide the correct base path.
    # The original script used base_path = "../" because it was inside scripts/.
    # If we run from root, base_path should be ".".
    
    generate_dataframe(args.base_path, args.output)

if __name__ == "__main__":
    main()
