#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def sort_with_sample(arr: np.ndarray, num_by_sample: list):
    """
    Sorts atoms within each sample by value.
    Matches logic in get_result_unlabel.py
    """
    sorted_arr = []
    index = 0
    for num in num_by_sample:
        # Sort the slice corresponding to the sample
        sorted_arr.append(np.sort(arr[index:index + num]))
        index += num
    return np.concatenate(sorted_arr)

def get_result(filename):
    """
    Reads pkl file and extracts flattened target, predict, and per-sample atom counts.
    Matches logic in get_result_unlabel.py
    """
    print(f"Loading {filename}...")
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    predict = []
    target = []
    num_by_sample = []
    
    for item in data:
        # Flatten batch
        predict.extend(item['predict'].reshape(-1).tolist())
        target.extend(item['target'].reshape(-1).tolist())
        
        # Calculate valid atoms per sample in the batch
        bsz = len(item['matid'])
        # select_atom is [batch_size, max_atoms]
        # sum(1) gives number of valid atoms per sample
        num_by_sample.extend(item['select_atom'].reshape(bsz, -1).sum(1).tolist())
        
    return target, predict, num_by_sample

def main():
    parser = argparse.ArgumentParser(description="Compare prediction errors from two PKL files.")
    parser.add_argument('--pkl1', required=True, help='Path to first .out.pkl file')
    parser.add_argument('--pkl2', required=True, help='Path to second .out.pkl file')
    parser.add_argument('--name1', default='Model 1', help='Label for first model')
    parser.add_argument('--name2', default='Model 2', help='Label for second model')
    parser.add_argument('--output', default='error_comparison_scatter.png', help='Output path for scatter plot')
    parser.add_argument('--title', default=None, help='Plot title')
    parser.add_argument('--symmetric-limit', type=float, default=None, help='Symmetric limit for x and y axes (e.g., 1.0 for [-1,1]). If not set, uses data range.')
    args = parser.parse_args()

    # 1. Load Data
    t1, p1, n1 = get_result(args.pkl1)
    t2, p2, n2 = get_result(args.pkl2)

    # 2. Sort atoms within each sample (Alignment)
    # Note: We convert to numpy array for sorting
    t1_sorted = sort_with_sample(np.array(t1), n1)
    p1_sorted = sort_with_sample(np.array(p1), n1)
    
    t2_sorted = sort_with_sample(np.array(t2), n2)
    p2_sorted = sort_with_sample(np.array(p2), n2)

    # 3. Verify Targets Match
    # Since we sorted targets, if the datasets are the same, t1_sorted should equal t2_sorted.
    if not np.array_equal(n1, n2):
        print("Error: Sample atom counts do not match! Are these the same dataset?")
        return

    if not np.allclose(t1_sorted, t2_sorted, atol=1e-5):
        print("Error: Sorted targets do not match! The datasets might be different or ordered differently.")
        max_diff = np.max(np.abs(t1_sorted - t2_sorted))
        print(f"Max difference in targets: {max_diff}")
        return
    
    print("Verification passed: Targets match.")
    
    # Use t1_sorted as the ground truth
    target = t1_sorted
    
    # 4. Calculate Errors (Target - Predict)
    error1 = target - p1_sorted
    error2 = target - p2_sorted
    
    # 5. Metrics
    mae1 = mean_absolute_error(target, p1_sorted)
    mae2 = mean_absolute_error(target, p2_sorted)
    rmse1 = np.sqrt(mean_squared_error(target, p1_sorted))
    rmse2 = np.sqrt(mean_squared_error(target, p2_sorted))
    
    print(f"\n--- Metrics ---")
    print(f"{args.name1}: MAE={mae1:.4f}, RMSE={rmse1:.4f}")
    print(f"{args.name2}: MAE={mae2:.4f}, RMSE={rmse2:.4f}")
    
    # 6. Plot Scatter
    plt.figure(figsize=(8, 8))
    plt.scatter(error1, error2, alpha=0.5, s=10, c='blue', edgecolors='none')
    
    # Draw y=x line (where errors are equal)
    # Get limits
    min_val = min(np.min(error1), np.min(error2))
    max_val = max(np.max(error1), np.max(error2)) 
    buffer = (max_val - min_val) * 0.05
    lims = [min_val - buffer, max_val + buffer]
    
    plt.plot(lims, lims, 'k--', alpha=0.75)
    plt.plot(lims, [-x for x in lims], 'k:', alpha=0.75)
    
    # Draw x=0 and y=0 lines for reference
    plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(0, color='gray', linestyle=':', alpha=0.5)

    plt.xlim(lims)
    plt.ylim(lims)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Apply symmetric limit if specified
    if args.symmetric_limit is not None:
        lim = args.symmetric_limit
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
    
    plt.xlabel(f'{args.name1} Error (Target - Predict)')
    plt.ylabel(f'{args.name2} Error (Target - Predict)')
    
    title = args.title if args.title else f'Prediction Error Comparison\n{args.name1} (MAE={mae1:.4f}) vs {args.name2} (MAE={mae2:.4f})'
    plt.title(title)
    #plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig(args.output, dpi=300)
    print(f"\nScatter plot saved to: {args.output}")

if __name__ == '__main__':
    main()
