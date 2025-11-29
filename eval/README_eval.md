# NMRNet++ Inference and Evaluation Tools

This folder contains tools for building evaluation datasets, running inference, and analyzing results for NMRNet++.

## Files Overview

1.  `rebuild_conflicts_aligned.py`: Identifies molecules with multiple solvent records in the validation set and generates a mapping file.
2.  `build_dual_solvent_subsets.py`: Builds paired datasets for counterfactual analysis (e.g., CDCl3 vs DMSO-d6) using the mapping.
3.  `run_infer_multisplit.sh`: Runs inference on the generated subsets using different model configurations.
4.  `compare_errors_scatter.py`: Visualizes prediction errors between two models/configurations.
5.  `eval_unlabel_from_merged.py`: Helper to split a merged LMDB into solvent-specific LMDBs.

## Usage Guide

### 1. Identify Paired Samples
First, find molecules that exist in multiple solvents (e.g., CDCl3 and DMSO-d6) within your validation set.

```bash
python eval/rebuild_conflicts_aligned.py \
    --data-root data/NMRexp_v0905/C \
    --output data/NMRexp_v0905/C/conflicts_valid.json
```

### 2. Build Paired Subsets
Generate paired LMDB datasets for a specific solvent pair (e.g., CDCl3 vs DMSO-d6). This creates three versions for each molecule:
- `label_a`: Original solvent A (e.g., CDCl3)
- `label_b`: Counterfactual solvent B (e.g., DMSO-d6)
- `label_base`: No solvent (Baseline)

```bash
python eval/build_dual_solvent_subsets.py \
    --mapping data/NMRexp_v0905/C/conflicts_valid.json \
    --original-lmdb data/NMRexp_v0905/C/valid.lmdb \
    --solvent-a CDCl3 \
    --solvent-b DMSO-d6 \
    --out-dir results/paired_intersection_C/CvD \
    --dict-path data/nmrshiftdb2_2018/All/dict.txt
```

### 3. Run Inference
Run inference on the generated subsets (`label_a`, `label_b`, `label_base`). The script supports a `mixed` mode to use different configurations for different subsets.

**Note:** Edit `run_infer_multisplit.sh` to set your `SUBSET_PARENT_DIR`, `SAVE_DIR` (Config A), and `SAVE_DIR_SECOND` (Config B) paths before running. `SUBSET_PARENT_DIR` should point to the directory containing the subsets (e.g., `results/paired_intersection_C/DvN`).

```bash
# Usage: MODE=[MODE] bash eval/run_infer_multisplit.sh
# MODE: mixed (default), first, or second
MODE=mixed bash eval/run_infer_multisplit.sh
```

### 4. Analyze Solvent Effect
Compare the prediction shifts when injecting solvents or not. Classified by solvent label.

**Step 1: Split Dataset by Solvent**
Split the merged validation set into solvent-specific subsets (CDCl3, DMSO-d6, OTHER).

```bash
python eval/split_unlabel_from_merged.py \
    --valid-path data/NMRexp_v0905/C/valid.lmdb \
    --out-root results/solvent_split_C \
    --dict-path data/NMRexp_v0905/C/dict.txt \
    --overwrite
```

**Step 2: Run Inference (Two Passes)**
Run inference twice on the split datasets: once with the solvent model (Config A) and once with the baseline model (Config B).

1.  **Edit `run_infer_multisplit.sh`**:
    -   Set `SUBSET_PARENT_DIR="results/solvent_split_C"`
    -   Set `SAVE_DIR="results/solvent_effect_C/with_solvent"` (Config A)
    -   Set `SAVE_DIR_SECOND="results/solvent_effect_C/no_solvent"` (Config B)

2.  **Run Pass 1 (With Solvent)**:
    ```bash
    MODE=first bash eval/run_infer_multisplit.sh
    ```

3.  **Run Pass 2 (Without Solvent)**:
    ```bash
    MODE=second bash eval/run_infer_multisplit.sh
    ```

**Step 3: Compare Results**
Generate a scatter plot to compare the errors of the two models for a specific solvent (e.g., CDCl3). Use `--symmetric-limit` to truncate the axes (e.g., to 20 ppm).

**Note:** The inference script generates `.pkl` files (e.g., `.../cv_seed_42_fold_1/cv_seed_42_fold_1_CDCl3.out.pkl`). You need to use `compare_errors_scatter.py` which supports reading these pickle files directly.

```bash
python eval/compare_errors_scatter.py \
    --pkl1 results/solvent_effect_C/with_solvent/cv_seed_42_fold_1/cv_seed_42_fold_1_CDCl3.out.pkl \
    --pkl2 results/solvent_effect_C/no_solvent/cv_seed_42_fold_1/cv_seed_42_fold_1_CDCl3.out.pkl \
    --title "Solvent Effect: CDCl3 (With vs Without)" \
    --output comparison_CDCl3.png \
    --symmetric-limit 20
```

