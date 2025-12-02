import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import os

def create_stratified_subset(input_csv, output_csv, n_samples=1000):
    """
    Creates a stratified subset of the input dataframe based on 'fire' and 'smoke' columns.
    """
    if not os.path.exists(input_csv):
        print(f"Error: Input file '{input_csv}' not found.")
        return

    df = pd.read_csv(input_csv)
    
    # Create a combined label for stratification
    df['stratify_col'] = df['fire'].astype(str) + "_" + df['smoke'].astype(str)
    
    # Check if we have enough samples
    if len(df) < n_samples:
        print(f"Warning: Input dataframe has fewer than {n_samples} rows. Returning full dataframe.")
        df.drop(columns=['stratify_col'], inplace=True)
        df.to_csv(output_csv, index=False)
        return

    # Stratified sampling
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=42)
    
    try:
        for train_index, _ in sss.split(df, df['stratify_col']):
            subset_df = df.iloc[train_index]
            break
    except ValueError as e:
         print(f"Error during stratification: {e}. Fallback to random sampling.")
         subset_df = df.sample(n=n_samples, random_state=42)

    # Clean up temporary column
    if 'stratify_col' in subset_df.columns:
        subset_df = subset_df.drop(columns=['stratify_col'])
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    subset_df.to_csv(output_csv, index=False)
    print(f"Successfully created subset with {len(subset_df)} rows at '{output_csv}'.")
    
    # Print distribution comparison
    print("\nOriginal Distribution:")
    print(df[['fire', 'smoke']].value_counts(normalize=True))
    print("\nSubset Distribution:")
    print(subset_df[['fire', 'smoke']].value_counts(normalize=True))

if __name__ == "__main__":
    input_path = 'dataframes/frame_labels_backup.csv'
    output_path = 'dataframes/prototype.csv'
    create_stratified_subset(input_path, output_path)
