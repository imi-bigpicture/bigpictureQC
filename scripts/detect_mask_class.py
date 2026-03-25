#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import multiresolutionimageinterface as mir

def read_lowres_mask(mask_path):
    if mir is None:
        raise RuntimeError("multiresolutionimageinterface is not available")

    reader = mir.MultiResolutionImageReader()
    img = reader.open(mask_path)
    if img is None:
        raise RuntimeError(f"Could not open mask: {mask_path}")

    # Read the lowest-resolution level only: enough to inspect label values
    level = img.getNumberOfLevels() - 1
    width, height = img.getLevelDimensions(level)
    patch = img.getUCharPatch(0, 0, width, height, level)

    # Handle MIR output shapes like HxWx1 or HxW
    arr = np.asarray(patch)
    if arr.ndim == 3:
        arr = arr[..., 0]

    return arr

def detect_mask_class(mask_path):
    arr = read_lowres_mask(mask_path)
    unique = np.unique(arr)

    # ignore possible empty/background padding values if present
    unique_set = set(int(v) for v in unique.tolist())

    if unique_set.issubset({0, 1}) and 1 in unique_set:
        return 1
    elif unique_set.issubset({1, 2}) and 2 in unique_set:
        return 2
    else:
        raise ValueError(
            f"Unsupported tissue mask values in {mask_path}: {sorted(unique_set)}. "
            "Expected {0,1} or {1,2}."
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", required=True, help="Path to tissue mask")
    args = parser.parse_args()

    mask_class = detect_mask_class(args.mask)
    print(mask_class)

if __name__ == "__main__":
    main()