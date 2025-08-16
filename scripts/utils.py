from pathlib import Path
import pandas as pd
import re
import numpy as np


def load_dataset(path:str, sep:None)->pd.DataFrame:
    return pd.read_csv(path, sep=sep)

def load_seq_(path:str):
    with open(path, 'r') as f:
        file = f.readlines()
        seq = ''.join([s.strip('\n') for s in file[1:]])
        mapping_db_seq = {str(i):i+1 for i in range(len(seq))}
        return seq, mapping_db_seq


def parse_mutation(mutation_str:str):
    match = re.match(r'^S([A-Za-z])(\d+)(.+)$', mutation_str)
    if match:
        return match.groups()
    else:
        raise ValueError(f"Invalid mutation format: {mutation_str}")

def mutate_seq(src:str, idx:str, dest:str, seq:str, mapping:dict, mapping_db_seq)->str:
    idx = mapping_db_seq[idx]
    new_seq =  seq[:idx] + dest + seq[idx + 1:]
    return new_seq

def mutate_sequence(mutation_string, seq, mapping_db_seq):
    if pd.isna(mutation_string):
        return None
    try:
        mutations = mutation_string.split(':')
        for m in mutations:
            src, idx, dest = parse_mutation(m)
            if idx in mapping_db_seq:
                mapped_idx = mapping_db_seq[idx]
                mutated_seq = seq[:mapped_idx] + dest + seq[mapped_idx + 1:]
                return mutated_seq
    except Exception:
        return None
    return None
def rotate_points(P, Q):
    """Rotates a set of of points using the Kabsch algorithm"""
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    D = np.linalg.det(V @ U.T)
    E = np.array([[1, 0, 0], [0, 1, 0], [0, 0, D]])
    R = V @ E @ U.T
    Pnew = (R @ P.T).T
    return Pnew


### Returns a np.ndarray of positions where two sequences differ
def get_mutation_position(s1, s2):
    """Get the position of mutations between two sequences"""
    return np.where(np.array(list(s1)) != np.array(list(s2)))[0]


def get_shared_indices(idx1, idx2):
    """Get the intersection between two sets of indices"""
    i1 = np.where(np.in1d(idx1, idx2))[0]
    i2 = np.where(np.in1d(idx2, idx1))[0]
    return i1, i2