#!/usr/bin/env python3
"""
Post-process artifact masks to compute and save quality metrics.

This script processes artifact segmentation masks and generates quality reports
containing per-slide metrics like quality score and artifact percentages.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Add source to path
sys.path.insert(0, '/home/user/source')
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from bigpictureQC.utils.metrics_writer import compute_and_save_metrics


def process_single_mask(
    mask_path: str,
    slide_id: Optional[str] = None,
    output_path: str = None,
    output_format: str = 'yaml',
    append: bool = True,
    wsi_level_preference: str = "lowest"
):
    """Process a single mask file and compute metrics."""
    mask_path = Path(mask_path)
    
    # Extract slide ID from filename if not provided
    if slide_id is None:
        slide_id = mask_path.stem.replace('_artifact', '').replace('_mask', '')
    
    # Set default output path if not provided
    if output_path is None:
        output_path = mask_path.parent / f'artifact_metrics.{output_format}'
    
    print(f"Processing: {slide_id}")
    print(f"  Mask: {mask_path}")
    
    metrics = compute_and_save_metrics(
        mask_path=str(mask_path),
        slide_id=slide_id,
        output_path=output_path,
        output_format=output_format,
        append=append,
        wsi_level_preference=wsi_level_preference,
    )

    # Print summary
    if metrics.get('error'):
        print(f"  ERROR: {metrics['error']}")
    else:
        print(f"  Artifact proportion: {metrics['artifact_proportion']:.4f}")
        print(f"  Quality Score: {metrics['quality_score']:.4f}")
        print(f"  Quality Score (excl. ink): {metrics['quality_score_excl_ink']:.4f}")
        print(f"  Total tissue pixels: {metrics['total_tissue_pixels']:,}")
        print("  Artifact proportions (fraction of tissue):")
        for name, value in metrics.get('artifact_proportions', {}).items():
            print(f"    {name:<13} {value:.4f}")
    
    return metrics


def process_folder(
    input_folder: str,
    output_path: str = None,
    output_format: str = 'yaml',
    pattern: str = '*_artifact.tif',
    wsi_level_preference: str = "lowest"
):
    """Process all mask files in a folder."""
    input_folder = Path(input_folder)
    
    if not input_folder.exists():
        print(f"ERROR: Input folder does not exist: {input_folder}")
        return
    
    # Set default output path
    if output_path is None:
        output_path = input_folder / f'artifact_metrics.{output_format}'
    
    # Find all mask files
    mask_files = sorted(input_folder.glob(pattern))
    
    if not mask_files:
        print(f"WARNING: No mask files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(mask_files)} mask files in {input_folder}")
    print(f"Output will be saved to: {output_path}")
    print("-" * 80)
    
    # Process each mask
    for i, mask_path in enumerate(mask_files, 1):
        print(f"\n[{i}/{len(mask_files)}]")
        try:
            process_single_mask(
                mask_path=str(mask_path),
                output_path=output_path,
                output_format=output_format,
                append=True,  # Always append in batch mode
                wsi_level_preference=wsi_level_preference
            )
        except Exception as e:
            print(f"  ERROR processing {mask_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print(f"Processing complete! Metrics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute quality metrics from artifact segmentation masks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single mask
  python compute_artifact_metrics.py --mask /path/to/slide_artifact.tif --output metrics.yaml
  
  # Process all masks in a folder
  python compute_artifact_metrics.py --folder /path/to/masks/ --output metrics.yaml
  
  # Use JSON format instead of YAML
  python compute_artifact_metrics.py --folder /path/to/masks/ --format json
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--mask', type=str,
                             help='Path to a single segmentation mask file')
    input_group.add_argument('--folder', type=str,
                             help='Path to folder containing multiple mask files')
    
    # Output options
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output metrics file (default: artifact_metrics.yaml in same folder)')
    parser.add_argument('--format', type=str, default='yaml', choices=['yaml', 'json'],
                        help='Output format (default: yaml)')
    
    # Processing options
    parser.add_argument('--slide_id', type=str, default=None,
                        help='Slide identifier (only for --mask mode, auto-extracted if not provided)')
    parser.add_argument('--pattern', type=str, default='*_artifact.tif',
                        help='File pattern for folder mode (default: *_artifact.tif)')
    parser.add_argument('--no-append', action='store_true',
                        help='Overwrite output file instead of appending (only for --mask mode)')
    parser.add_argument('--wsi_level_preference', type=str, default='lowest',
                        choices=['lowest', 'highest'],
                        help='WSI pyramid level preference: "lowest" (fastest) or "highest" (full resolution)')
    
    args = parser.parse_args()
    
    # Process based on mode
    if args.mask:
        process_single_mask(
            mask_path=args.mask,
            slide_id=args.slide_id,
            output_path=args.output,
            output_format=args.format,
            append=not args.no_append,
            wsi_level_preference=args.wsi_level_preference
        )
    else:
        process_folder(
            input_folder=args.folder,
            output_path=args.output,
            output_format=args.format,
            pattern=args.pattern,
            wsi_level_preference=args.wsi_level_preference
        )


if __name__ == '__main__':
    main()
