#!/usr/bin/env python3
"""
Test script for balanced sampling function
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import from scripts
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from scripts.utils import load_dataset, load_seq_
from scripts.balanced_sampling import generate_idx_for_n_mutations, get_numb_mut


def main():
    """Test the balanced sampling function with your dataset"""
    
    # Load your dataset (adjust paths as needed)
    data_set_path = Path('data/amino_acid_genotypes_to_brightness.tsv')
    original_seq_path = Path('data/P42212.fasta.txt')
    
    print("Loading dataset...")
    dataset = load_dataset(str(data_set_path), sep='\t')
    seq, mapping = load_seq_(str(original_seq_path))
    
    # Add num_mut column
    print("Adding num_mut column...")
    dataset['num_mut'] = dataset['aaMutations'].apply(lambda mut: get_numb_mut(mut))
    
    print(f"Dataset shape: {dataset.shape}")
    print(f"Original num_mut distribution:")
    print(dataset['num_mut'].value_counts().sort_index())
    
    # Generate balanced subset
    n_samples = min(1000, len(dataset) // 2)  # Use half the dataset or 1000, whichever is smaller
    print(f"\nGenerating balanced subset with {n_samples} samples...")
    
    balanced_subset = generate_idx_for_n_mutations(
        dataset, 
        n=n_samples, 
        output_file='balanced_subset.txt'
    )
    
    print(f"\nBalanced subset shape: {balanced_subset.shape}")
    print("Done! Check 'balanced_subset.txt' for the results.")


if __name__ == "__main__":
    main() 