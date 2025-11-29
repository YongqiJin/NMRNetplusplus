#!/usr/bin/env bash
set -euo pipefail

# run_infer_multisplit.sh
# Enhanced wrapper for multi-split inference with dual configuration support.
#
# Usage (positional style similar to infer_all_with_solv.sh):
#   ./run_infer_multisplit.sh <save_dir> <element> <arch> <solvent_flags>
# Example:
#   ./run_infer_multisplit.sh /path/to/save_dir H unimol_large_solv "--solvent-embed-before-backbone True --bos-only"
#
# Enhanced features:
# - Support for dual configurations (first/second) and mixed mode.
# - MODE: "first" (use first config), "second" (use second config), "mixed" (use second for label_base, first otherwise)
# - Set MODE, SAVE_DIR_SECOND, SOLVENT_FLAGS_SECOND via environment variables.

INFER_PY="./uninmr/infer.py"
GET_RESULT_UNLABEL_PY="./uninmr/utils/get_result_unlabel.py"

if [ ! -f "$INFER_PY" ]; then
  echo "Cannot find infer script at $INFER_PY" >&2
  exit 2
fi

# First configuration (with solvent injection)
SAVE_DIR="./results/triclass_infer_C"
ELEMENT="C"
ARCH="unimol_large_solv_v2"
SOLVENT_FLAGS="--solvent-embed-before-backbone --bos-only  --solvent-max-types 4"

# Second configuration (baseline)
SAVE_DIR_SECOND="${SAVE_DIR_SECOND:-./results/triclass_blank_C}"
SOLVENT_FLAGS_SECOND="${SOLVENT_FLAGS_SECOND:---solvent-max-types 0}"

# Mode: first, second, or mixed
MODE="${MODE:-first}"

# Defaults - adjust if needed
DATA_PATH="./data/nmrshiftdb2_2018/All"
UNLABELED_DATA_PATH="./data/NMRexp_v0905/${ELEMENT}"
NWORKERS=8
DDP_BACKEND="c10d"
BATCH_SIZE=64
N_FOLDS=1
CV_SEED=42
DICT_NAME="dict.txt"
SUBSET_PARENT_DIR="/mnt/fs_mol/wyc/NMRNet++/results/paired_intersection_C/DvN"
mkdir -p "${SAVE_DIR}"

# Function to select config based on mode and subset
select_config() {
    local subset_name="$1"
    case "$MODE" in
        "first")
            echo "$SAVE_DIR" "$SOLVENT_FLAGS"
            ;;
        "second")
            echo "$SAVE_DIR_SECOND" "$SOLVENT_FLAGS_SECOND"
            ;;
        "mixed")
            if [[ "$subset_name" == "label_base" ]]; then
                echo "$SAVE_DIR_SECOND" "$SOLVENT_FLAGS_SECOND"
            else
                echo "$SAVE_DIR" "$SOLVENT_FLAGS"
            fi
            ;;
        *)
            echo "Invalid MODE: $MODE. Use first, second, or mixed." >&2
            exit 1
            ;;
    esac
}

# Optionally, if you have a subset folder under SUBSET_PARENT_DIR, run inference on it
if [ -n "$SUBSET_PARENT_DIR" ] && [ -d "$SUBSET_PARENT_DIR" ]; then
  for subset_dir in "$SUBSET_PARENT_DIR"/*; do
    if [ -d "$subset_dir" ]; then
      subset_name="$(basename "$subset_dir")"
      printf "\n=== Running inference on subset: %s (MODE: %s) ===\n" "$subset_name" "$MODE"
      
      # Select config
      read -r selected_save_dir selected_solvent_flags <<< "$(select_config "$subset_name")"
      printf "Using SAVE_DIR: %s\n" "$selected_save_dir"
      printf "Using SOLVENT_FLAGS: %s\n" "$selected_solvent_flags"
      
      mkdir -p "$selected_save_dir"

      for fold in $(seq 1 $(($N_FOLDS))); do
        out_dir="${selected_save_dir}/cv_seed_${CV_SEED}_fold_${fold}"
        mkdir -p "$out_dir"
        python3 "$INFER_PY" --user-dir ./uninmr "$subset_dir" --valid-subset valid --subset_name "$subset_name" \
          --results-path "$out_dir" --saved-dir "$selected_save_dir" \
          --num-workers $NWORKERS --ddp-backend=$DDP_BACKEND --batch-size $BATCH_SIZE \
          --task uninmr_solv --loss 'atom_regloss_mae' --arch "$ARCH" \
          --dict-name "$DICT_NAME" \
          $selected_solvent_flags \
          --path ${selected_save_dir}/cv_seed_${CV_SEED}_fold_${fold}/checkpoint_last.pt \
          --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
          --log-interval 50 --log-format simple --required-batch-size-multiple 1 \
          --selected-atom ${ELEMENT} --gaussian-kernel --atom-descriptor 0 --split-mode infer || true
      done

      # Per-subset evaluation using get_result_unlabel.py (cv mode), targeting files named with subset_name
      if [ -f "$GET_RESULT_UNLABEL_PY" ]; then
        DICT_FOR_SUBSET="${subset_dir}/dict.txt"
        if [ ! -f "$DICT_FOR_SUBSET" ]; then
          # Fallback to the original dict under DATA_PATH
          DICT_FOR_SUBSET="${DATA_PATH}/${DICT_NAME}"
        fi
        echo "Running get_result_unlabel.py for subset $subset_name with dict: $DICT_FOR_SUBSET"
        python3 "$GET_RESULT_UNLABEL_PY" \
          --path "$selected_save_dir" \
          --file_end "*${subset_name}.out.pkl" \
          --mode cv \
          --dict "$DICT_FOR_SUBSET" \
          2>&1 | tee "${selected_save_dir}/result_unlabel_subset_${subset_name}.log" || true
      else
        echo "Warning: $GET_RESULT_UNLABEL_PY not found; skipping subset evaluation for $subset_name" >&2
      fi

      echo "Completed subset: $subset_dir"
    fi
  done
else
  echo "No subset parent dir found at: $SUBSET_PARENT_DIR (skipping subset inference)"
fi

echo "All inference runs submitted. Check logs under $SAVE_DIR (and $SAVE_DIR_SECOND if used)"
