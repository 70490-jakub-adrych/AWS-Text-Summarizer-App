"""
This script downloads and prepares the CNN/DailyMail dataset for SageMaker training.
It runs as a SageMaker Processing job to prepare the data.
"""
import logging
import os
import argparse
import sys
import subprocess

# Install required packages at runtime
print("Installing required packages...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "pandas", "transformers"])

# Now we can import the required modules
import pandas as pd
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-split-size", type=float, default=0.1)  # Reduced to 10% for faster processing
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directories
    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "validation"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "test"), exist_ok=True)
    
    logger.info("Downloading CNN/DailyMail dataset...")
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    
    # We'll use a smaller subset for faster processing
    train_size = int(len(dataset['train']) * args.train_split_size)
    val_size = int(len(dataset['validation']) * args.train_split_size)
    test_size = int(len(dataset['test']) * args.train_split_size)
    
    logger.info(f"Processing training split (using {train_size} samples)...")
    train_df = pd.DataFrame({
        "article": dataset["train"]["article"][:train_size],
        "highlights": dataset["train"]["highlights"][:train_size]
    })
    train_df.to_csv(os.path.join(args.output_dir, "train", "train.csv"), index=False)
    
    logger.info(f"Processing validation split (using {val_size} samples)...")
    val_df = pd.DataFrame({
        "article": dataset["validation"]["article"][:val_size],
        "highlights": dataset["validation"]["highlights"][:val_size]
    })
    val_df.to_csv(os.path.join(args.output_dir, "validation", "validation.csv"), index=False)
    
    logger.info(f"Processing test split (using {test_size} samples)...")
    test_df = pd.DataFrame({
        "article": dataset["test"]["article"][:test_size],
        "highlights": dataset["test"]["highlights"][:test_size]
    })
    test_df.to_csv(os.path.join(args.output_dir, "test", "test.csv"), index=False)
    
    logger.info("Data processing completed successfully!")

if __name__ == "__main__":
    main()
