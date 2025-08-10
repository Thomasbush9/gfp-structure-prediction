import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy


def calculate_position_entropy(sequences: List[str], 
                             position: int, 
                             gap_char: str = '-',
                             method: str = 'shannon') -> float:
    """
    Calculate entropy for a specific position in a sequence alignment.
    
    Args:
        sequences: List of aligned sequences
        position: Position index (0-based)
        gap_char: Character representing gaps in alignment
        method: Entropy calculation method ('shannon', 'gini', 'simpson')
    
    Returns:
        Entropy value for the position
    """
    if position >= len(sequences[0]):
        raise ValueError(f"Position {position} is out of range for sequences of length {len(sequences[0])}")
    
    # Extract amino acids at this position
    position_aa = [seq[position] for seq in sequences if position < len(seq)]
    
    # Filter out gaps if specified
    if gap_char:
        position_aa = [aa for aa in position_aa if aa != gap_char]
    
    if not position_aa:
        return 0.0
    
    # Count amino acid frequencies
    aa_counts = Counter(position_aa)
    total_count = len(position_aa)
    
    if method == 'shannon':
        # Shannon entropy: -sum(p_i * log2(p_i))
        probabilities = [count / total_count for count in aa_counts.values()]
        return -sum(p * np.log2(p) for p in probabilities if p > 0)
    
    elif method == 'gini':
        # Gini impurity: 1 - sum(p_i^2)
        probabilities = [count / total_count for count in aa_counts.values()]
        return 1 - sum(p**2 for p in probabilities)
    
    elif method == 'simpson':
        # Simpson's diversity: 1 - sum(p_i^2)
        probabilities = [count / total_count for count in aa_counts.values()]
        return 1 - sum(p**2 for p in probabilities)
    
    else:
        raise ValueError(f"Unknown entropy method: {method}")


def calculate_conservation_entropy(sequences: List[str], 
                                 gap_char: str = '-',
                                 method: str = 'shannon',
                                 normalize: bool = True) -> Dict[int, float]:
    """
    Calculate conservation entropy for all positions in a sequence alignment.
    
    Args:
        sequences: List of aligned sequences
        gap_char: Character representing gaps in alignment
        method: Entropy calculation method
        normalize: Whether to normalize entropy by maximum possible entropy
    
    Returns:
        Dictionary mapping position index to entropy value
    """
    if not sequences:
        return {}
    
    max_length = max(len(seq) for seq in sequences)
    position_entropies = {}
    
    for position in range(max_length):
        entropy_val = calculate_position_entropy(sequences, position, gap_char, method)
        
        if normalize and method == 'shannon':
            # Normalize by maximum possible entropy (log2 of number of unique amino acids)
            unique_aas = set()
            for seq in sequences:
                if position < len(seq) and seq[position] != gap_char:
                    unique_aas.add(seq[position])
            
            max_entropy = np.log2(len(unique_aas)) if len(unique_aas) > 1 else 0
            if max_entropy > 0:
                entropy_val = entropy_val / max_entropy
        
        position_entropies[position] = entropy_val
    
    return position_entropies


def calculate_amino_acid_frequencies(sequences: List[str], 
                                   position: int,
                                   gap_char: str = '-') -> Dict[str, float]:
    """
    Calculate amino acid frequencies at a specific position.
    
    Args:
        sequences: List of aligned sequences
        position: Position index (0-based)
        gap_char: Character representing gaps in alignment
    
    Returns:
        Dictionary mapping amino acid to frequency
    """
    if position >= len(sequences[0]):
        raise ValueError(f"Position {position} is out of range")
    
    position_aa = [seq[position] for seq in sequences if position < len(seq)]
    position_aa = [aa for aa in position_aa if aa != gap_char]
    
    if not position_aa:
        return {}
    
    aa_counts = Counter(position_aa)
    total_count = len(position_aa)
    
    return {aa: count / total_count for aa, count in aa_counts.items()}


def analyze_conservation_patterns(sequences: List[str],
                                gap_char: str = '-',
                                method: str = 'shannon') -> pd.DataFrame:
    """
    Comprehensive analysis of conservation patterns across all positions.
    
    Args:
        sequences: List of aligned sequences
        gap_char: Character representing gaps in alignment
        method: Entropy calculation method
    
    Returns:
        DataFrame with conservation analysis for each position
    """
    if not sequences:
        return pd.DataFrame()
    
    max_length = max(len(seq) for seq in sequences)
    analysis_data = []
    
    for position in range(max_length):
        # Calculate entropy
        entropy_val = calculate_position_entropy(sequences, position, gap_char, method)
        
        # Calculate amino acid frequencies
        aa_freqs = calculate_amino_acid_frequencies(sequences, position, gap_char)
        
        # Find most common amino acid
        most_common_aa = max(aa_freqs.items(), key=lambda x: x[1]) if aa_freqs else (None, 0)
        
        # Calculate coverage (non-gap sequences)
        coverage = sum(1 for seq in sequences 
                      if position < len(seq) and seq[position] != gap_char) / len(sequences)
        
        analysis_data.append({
            'position': position,
            'entropy': entropy_val,
            'most_common_aa': most_common_aa[0],
            'most_common_freq': most_common_aa[1],
            'coverage': coverage,
            'unique_aas': len(aa_freqs),
            'aa_frequencies': aa_freqs
        })
    
    return pd.DataFrame(analysis_data)


def plot_conservation_entropy(entropy_dict: Dict[int, float],
                            title: str = "Conservation Entropy by Position",
                            figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot conservation entropy across all positions.
    
    Args:
        entropy_dict: Dictionary mapping position to entropy value
        title: Plot title
        figsize: Figure size
    """
    positions = sorted(entropy_dict.keys())
    entropy_values = [entropy_dict[pos] for pos in positions]
    
    plt.figure(figsize=figsize)
    plt.plot(positions, entropy_values, 'b-', linewidth=2)
    plt.fill_between(positions, entropy_values, alpha=0.3)
    plt.xlabel('Position')
    plt.ylabel('Entropy')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_conservation_heatmap(sequences: List[str],
                             gap_char: str = '-',
                             max_positions: int = 100,
                             figsize: Tuple[int, int] = (15, 8)) -> None:
    """
    Create a heatmap showing amino acid frequencies across positions.
    
    Args:
        sequences: List of aligned sequences
        gap_char: Character representing gaps in alignment
        max_positions: Maximum number of positions to display
        figsize: Figure size
    """
    if not sequences:
        return
    
    max_length = min(max(len(seq) for seq in sequences), max_positions)
    
    # Get all unique amino acids
    all_aas = set()
    for seq in sequences:
        for aa in seq[:max_length]:
            if aa != gap_char:
                all_aas.add(aa)
    
    # Create frequency matrix
    aa_list = sorted(all_aas)
    freq_matrix = np.zeros((len(aa_list), max_length))
    
    for pos in range(max_length):
        aa_freqs = calculate_amino_acid_frequencies(sequences, pos, gap_char)
        for i, aa in enumerate(aa_list):
            freq_matrix[i, pos] = aa_freqs.get(aa, 0)
    
    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(freq_matrix, 
                xticklabels=range(1, max_length + 1),
                yticklabels=aa_list,
                cmap='viridis',
                cbar_kws={'label': 'Frequency'})
    plt.xlabel('Position')
    plt.ylabel('Amino Acid')
    plt.title('Amino Acid Frequency Heatmap')
    plt.tight_layout()
    plt.show()


def identify_conserved_positions(entropy_dict: Dict[int, float],
                               threshold: float = 0.1,
                               method: str = 'shannon') -> List[int]:
    """
    Identify highly conserved positions based on entropy threshold.
    
    Args:
        entropy_dict: Dictionary mapping position to entropy value
        threshold: Entropy threshold for conservation (lower = more conserved)
        method: Entropy method used
    
    Returns:
        List of conserved position indices
    """
    if method == 'shannon':
        # For Shannon entropy, lower values indicate more conservation
        return [pos for pos, entropy_val in entropy_dict.items() if entropy_val <= threshold]
    else:
        # For Gini/Simpson, lower values indicate more conservation
        return [pos for pos, entropy_val in entropy_dict.items() if entropy_val <= threshold]


def calculate_conservation_score(sequences: List[str],
                               position: int,
                               gap_char: str = '-') -> float:
    """
    Calculate a simple conservation score (1 - entropy) for a position.
    
    Args:
        sequences: List of aligned sequences
        position: Position index (0-based)
        gap_char: Character representing gaps in alignment
    
    Returns:
        Conservation score (0 = no conservation, 1 = perfect conservation)
    """
    entropy_val = calculate_position_entropy(sequences, position, gap_char, 'shannon')
    
    # Normalize by maximum possible entropy for this position
    unique_aas = set()
    for seq in sequences:
        if position < len(seq) and seq[position] != gap_char:
            unique_aas.add(seq[position])
    
    max_entropy = np.log2(len(unique_aas)) if len(unique_aas) > 1 else 0
    
    if max_entropy == 0:
        return 1.0  # Perfectly conserved
    
    conservation_score = 1 - (entropy_val / max_entropy)
    return max(0, conservation_score)


# Example usage and testing
if __name__ == "__main__":
    # Example sequences (aligned)
    example_sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    ]
    
    # Calculate conservation entropy
    entropy_dict = calculate_conservation_entropy(example_sequences)
    
    # Analyze conservation patterns
    analysis_df = analyze_conservation_patterns(example_sequences)
    
    print("Conservation Analysis:")
    print(analysis_df.head())
    
    # Plot results
    plot_conservation_entropy(entropy_dict)
    
    # Identify conserved positions
    conserved_positions = identify_conserved_positions(entropy_dict, threshold=0.1)
    print(f"\nConserved positions (entropy <= 0.1): {conserved_positions[:10]}...") 