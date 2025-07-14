"""
Split a parquet file into train/test sets

python data_preprocess/split_parquet.py --input ./data/eurus2_sft_math/llama70b_sft_data_generation.parquet
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_parquet(input_file, output_dir=None, train_ratio=0.9, random_seed=42):
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {input_file}")
    df = pd.read_parquet(input_file)
    
    total_rows = len(df)
    print(f"Total number of examples: {total_rows}")
    
    train_df, test_df = train_test_split(
        df, 
        train_size=train_ratio,
        random_state=random_seed,
        shuffle=True
    )
    
    print(f"Training set size: {len(train_df)} ({len(train_df)/total_rows:.2%})")
    print(f"Test set size: {len(test_df)} ({len(test_df)/total_rows:.2%})")
    
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    
    train_output = os.path.join(output_dir, f"{base_filename}_train.parquet")
    test_output = os.path.join(output_dir, f"{base_filename}_test.parquet")
    
    train_df.to_parquet(train_output, index=False)
    test_df.to_parquet(test_output, index=False)
    
    print(f"Training set saved to: {train_output}")
    print(f"Test set saved to: {test_output}")
    
    return train_output, test_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a parquet file into train/test sets")
    parser.add_argument("--input", required=True, help="Input parquet file path")
    parser.add_argument("--output_dir", default=None, help="Output directory for split files (default: same as input file)")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Ratio for training set (default: 0.9)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    split_parquet(
        args.input,
        args.output_dir,
        train_ratio=args.train_ratio,
        random_seed=args.seed
    )