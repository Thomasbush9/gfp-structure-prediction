import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell
def _():
    import os
    from pathlib import Path
    from tqdm import tqdm
    import pandas as pd
    import xarray as xr
    import re
    import marimo as mo
    return Path, mo, os


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
    return


@app.cell
def _(Path):
    data_dir = Path('../Data')
    original_seq_path = data_dir / '/Users/thomasbush/Documents/ML/gfp_tryout/data/P42212.fasta.txt'
    data_set_path = data_dir / '/Users/thomasbush/Documents/ML/gfp_tryout/data/amino_acid_genotypes_to_brightness.tsv'
    return data_set_path, original_seq_path


@app.cell
def _(os):
    import sys
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

    # Step 2: Add it to sys.path
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    # Step 3: Import the function from utils.py
    from scripts.utils import mutate_sequence, load_seq_, load_dataset
    return load_dataset, load_seq_, mutate_sequence


@app.cell
def _(
    data_set_path,
    load_dataset,
    load_seq_,
    mutate_sequence,
    original_seq_path,
):
    dataset = load_dataset(data_set_path, sep='\t')
    seq, mapping = load_seq_(original_seq_path)
    dataset['seq_mutated'] = dataset['aaMutations'].apply(
        lambda muts: mutate_sequence(muts, seq=seq, mapping_db_seq=mapping)
    )
    return (dataset,)


@app.cell
def _():
    import numpy as np
    return


@app.cell
def _(dataset):
    dataset['lenght'] = dataset['aaMutations'][1:].apply(lambda muts: len(muts))

    return


@app.cell
def _(dataset):
    dataset[dataset['lenght'] > 10]
    return


@app.cell
def _(mo):
    mo.md(r"""### Number of mutation vs change in brightness """)
    return


app._unparsable_cell(
    r"""
    def get_number
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
