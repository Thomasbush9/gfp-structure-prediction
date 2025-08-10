import pandas as pd
import numpy as np
from pathlib import Path


def generate_idx_for_n_mutations(dataset, n, output_file='balanced_subset.txt'):
    """
    Generate a balanced subset of the dataset that maintains the original distribution
    of num_mut while selecting n samples (where n < N, total dataset size).
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        The input dataset with a 'num_mut' column
    n : int
        Number of samples to select (must be < len(dataset))
    output_file : str
        Path to the output .txt file
        
    Returns:
    --------
    pd.DataFrame
        The balanced subset with original indices preserved
    """
    # Validate input
    if n >= len(dataset):
        raise ValueError(f"n ({n}) must be less than dataset size ({len(dataset)})")
    
    # Calculate the original distribution of num_mut
    original_dist = dataset['num_mut'].value_counts(normalize=True)
    
    # Calculate how many samples to select from each num_mut category
    target_counts = (original_dist * n).round().astype(int)
    
    # Ensure we don't exceed the available samples in each category
    available_counts = dataset['num_mut'].value_counts()
    final_counts = {}
    
    for num_mut in target_counts.index:
        target_count = target_counts[num_mut]
        available_count = available_counts.get(num_mut, 0)
        final_counts[num_mut] = min(target_count, available_count)
    
    # Adjust if we have fewer samples than requested
    total_selected = sum(final_counts.values())
    if total_selected < n:
        # Distribute remaining samples proportionally
        remaining = n - total_selected
        for num_mut in sorted(original_dist.index, key=lambda x: original_dist[x], reverse=True):
            if remaining <= 0:
                break
            available = available_counts.get(num_mut, 0) - final_counts.get(num_mut, 0)
            to_add = min(remaining, available)
            final_counts[num_mut] = final_counts.get(num_mut, 0) + to_add
            remaining -= to_add
    
    # Sample from each category
    balanced_subset = []
    
    for num_mut, count in final_counts.items():
        if count > 0:
            # Get indices for this num_mut category
            category_indices = dataset[dataset['num_mut'] == num_mut].index
            # Randomly sample without replacement
            selected_indices = np.random.choice(category_indices, size=count, replace=False)
            balanced_subset.extend(selected_indices)
    
    # Create the balanced dataset
    balanced_dataset = dataset.loc[balanced_subset].copy()
    
    # Save to txt file with idx and n_mut
    output_data = []
    for idx in balanced_subset:
        n_mut = dataset.loc[idx, 'num_mut']
        output_data.append(f"{idx}\t{n_mut}")
    
    with open(output_file, 'w') as f:
        f.write("idx\tn_mut\n")
        for line in output_data:
            f.write(line + "\n")
    
    print(f"Balanced subset created with {len(balanced_dataset)} samples")
    print(f"Original distribution:")
    print(dataset['num_mut'].value_counts(normalize=True).sort_index())
    print(f"\nBalanced subset distribution:")
    print(balanced_dataset['num_mut'].value_counts(normalize=True).sort_index())
    print(f"\nResults saved to: {output_file}")
    
    return balanced_dataset


def get_numb_mut(mut: str) -> int:
    """Helper function to count number of mutations from mutation string"""
    if type(mut) == str:
        n = len(mut.split(':'))
    else: 
        return 0
    return n


# Example usage function
def example_usage():
    """
    Example of how to use the balanced sampling function
    """
    # Load your dataset (replace with your actual data loading)
    # dataset = load_dataset(data_set_path, sep='\t')
    
    # Add num_mut column if not already present
    # dataset['num_mut'] = dataset['aaMutations'].apply(lambda mut: get_numb_mut(mut))
    
    # Generate balanced subset
    # balanced_subset = generate_idx_for_n_mutations(dataset, n=1000, output_file='balanced_subset.txt')
    
    print("Example usage:")
    print("1. Load your dataset")
    print("2. Add num_mut column: dataset['num_mut'] = dataset['aaMutations'].apply(lambda mut: get_numb_mut(mut))")
    print("3. Call: balanced_subset = generate_idx_for_n_mutations(dataset, n=1000, output_file='balanced_subset.txt')")


if __name__ == "__main__":
    example_usage() 