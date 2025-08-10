#!/usr/bin/env python3
"""
Conservation Entropy Example

This script demonstrates how to calculate conservation entropy for protein sequences.
Conservation entropy measures the diversity of amino acids at each position in a 
sequence alignment. Lower entropy indicates more conserved positions.

Key Concepts:
1. Shannon Entropy: -sum(p_i * log2(p_i)) where p_i is the frequency of amino acid i
2. Normalized Entropy: Entropy divided by maximum possible entropy for that position
3. Conservation Score: 1 - (normalized entropy) - higher values indicate more conservation
"""

import numpy as np
from collections import Counter
from typing import List, Dict


def calculate_position_entropy(sequences: List[str], position: int) -> float:
    """
    Calculate Shannon entropy for a specific position in aligned sequences.
    
    Example:
        Position 0 in sequences: ['A', 'A', 'A', 'B', 'B']
        Frequencies: A=0.6, B=0.4
        Entropy = -(0.6 * log2(0.6) + 0.4 * log2(0.4)) = 0.971
    
    Args:
        sequences: List of aligned protein sequences
        position: Position index (0-based)
    
    Returns:
        Shannon entropy value
    """
    # Extract amino acids at this position
    position_aa = [seq[position] for seq in sequences]
    
    # Count frequencies
    aa_counts = Counter(position_aa)
    total_count = len(position_aa)
    
    # Calculate Shannon entropy
    entropy = 0.0
    for count in aa_counts.values():
        p = count / total_count
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy


def calculate_normalized_entropy(sequences: List[str], position: int) -> float:
    """
    Calculate normalized entropy (entropy / max_entropy).
    
    This normalizes by the maximum possible entropy for the number of unique
    amino acids at that position.
    """
    entropy = calculate_position_entropy(sequences, position)
    
    # Get unique amino acids at this position
    unique_aas = set(seq[position] for seq in sequences)
    max_entropy = np.log2(len(unique_aas)) if len(unique_aas) > 1 else 0
    
    if max_entropy == 0:
        return 0.0  # Perfectly conserved
    
    return entropy / max_entropy


def calculate_conservation_score(sequences: List[str], position: int) -> float:
    """
    Calculate conservation score (1 - normalized_entropy).
    
    Returns:
        0 = no conservation (high diversity)
        1 = perfect conservation (single amino acid)
    """
    normalized_entropy = calculate_normalized_entropy(sequences, position)
    return 1 - normalized_entropy


def analyze_conservation_example():
    """Demonstrate conservation entropy calculation with examples."""
    
    print("=== CONSERVATION ENTROPY ANALYSIS ===\n")
    
    # Example 1: Perfectly conserved position
    sequences1 = [
        "AAAAA",
        "AAAAA", 
        "AAAAA",
        "AAAAA",
        "AAAAA"
    ]
    
    print("Example 1: Perfectly Conserved Position")
    print("Sequences:", sequences1)
    entropy1 = calculate_position_entropy(sequences1, 0)
    norm_entropy1 = calculate_normalized_entropy(sequences1, 0)
    cons_score1 = calculate_conservation_score(sequences1, 0)
    
    print(f"Position 0: Entropy={entropy1:.3f}, Normalized={norm_entropy1:.3f}, Conservation={cons_score1:.3f}")
    print("→ All amino acids are the same, so entropy = 0 (perfectly conserved)\n")
    
    # Example 2: Mixed position
    sequences2 = [
        "AAAAA",
        "AAAAA",
        "ABAAA", 
        "AAAAA",
        "ACAAA"
    ]
    
    print("Example 2: Mixed Position")
    print("Sequences:", sequences2)
    entropy2 = calculate_position_entropy(sequences2, 0)
    norm_entropy2 = calculate_normalized_entropy(sequences2, 0)
    cons_score2 = calculate_conservation_score(sequences2, 0)
    
    print(f"Position 0: Entropy={entropy2:.3f}, Normalized={norm_entropy2:.3f}, Conservation={cons_score2:.3f}")
    print("→ Some diversity, moderate conservation\n")
    
    # Example 3: Highly variable position
    sequences3 = [
        "AAAAA",
        "BAAAA",
        "CAAAA",
        "DAAAA",
        "EAAAA"
    ]
    
    print("Example 3: Highly Variable Position")
    print("Sequences:", sequences3)
    entropy3 = calculate_position_entropy(sequences3, 0)
    norm_entropy3 = calculate_normalized_entropy(sequences3, 0)
    cons_score3 = calculate_conservation_score(sequences3, 0)
    
    print(f"Position 0: Entropy={entropy3:.3f}, Normalized={norm_entropy3:.3f}, Conservation={cons_score3:.3f}")
    print("→ High diversity, low conservation\n")
    
    # Example 4: Realistic protein sequences
    protein_sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    ]
    
    print("Example 4: Realistic Protein Sequences")
    print("Analyzing conservation across all positions...")
    
    # Calculate conservation for all positions
    conservation_scores = []
    for pos in range(len(protein_sequences[0])):
        score = calculate_conservation_score(protein_sequences, pos)
        conservation_scores.append(score)
    
    print(f"Average conservation score: {np.mean(conservation_scores):.3f}")
    print(f"Most conserved position: {np.argmax(conservation_scores)} (score: {np.max(conservation_scores):.3f})")
    print(f"Least conserved position: {np.argmin(conservation_scores)} (score: {np.min(conservation_scores):.3f})")


def demonstrate_with_mutations():
    """Demonstrate how mutations affect conservation entropy."""
    
    print("\n=== MUTATION IMPACT ON CONSERVATION ===\n")
    
    # Original sequence
    original = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    
    # Create mutated sequences
    mutated_sequences = [
        original,  # Original
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",  # No mutation
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",  # No mutation
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",  # No mutation
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",  # No mutation
    ]
    
    # Add some mutations at specific positions
    # Position 10: L -> A
    mutated_sequences[1] = mutated_sequences[1][:10] + "A" + mutated_sequences[1][11:]
    # Position 20: E -> D  
    mutated_sequences[2] = mutated_sequences[2][:20] + "D" + mutated_sequences[2][21:]
    # Position 30: K -> R
    mutated_sequences[3] = mutated_sequences[3][:30] + "R" + mutated_sequences[3][31:]
    # Position 40: V -> I
    mutated_sequences[4] = mutated_sequences[4][:40] + "I" + mutated_sequences[4][41:]
    
    print("Analyzing conservation with mutations at positions 10, 20, 30, 40...")
    
    # Calculate conservation for mutated positions
    mutation_positions = [10, 20, 30, 40]
    for pos in mutation_positions:
        score = calculate_conservation_score(mutated_sequences, pos)
        print(f"Position {pos}: Conservation Score = {score:.3f}")
    
    # Compare with non-mutated positions
    non_mutated_positions = [0, 5, 15, 25, 35, 45]
    print("\nNon-mutated positions:")
    for pos in non_mutated_positions:
        score = calculate_conservation_score(mutated_sequences, pos)
        print(f"Position {pos}: Conservation Score = {score:.3f}")


if __name__ == "__main__":
    analyze_conservation_example()
    demonstrate_with_mutations()
    
    print("\n=== KEY TAKEAWAYS ===")
    print("1. Conservation entropy measures amino acid diversity at each position")
    print("2. Lower entropy = more conserved = functionally important")
    print("3. Normalized entropy accounts for the number of unique amino acids")
    print("4. Conservation score = 1 - normalized_entropy (higher = more conserved)")
    print("5. This helps identify critical regions in protein evolution") 