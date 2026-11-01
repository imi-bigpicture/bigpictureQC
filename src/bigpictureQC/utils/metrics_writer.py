"""Utility for computing and saving quality metrics from artifact segmentation masks."""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime

def _to_builtin(obj):
    """Recursively convert numpy scalars/arrays to Python builtins for safe YAML/JSON."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]

    # numpy scalar -> python scalar
    if isinstance(obj, np.generic):
        return obj.item()

    # numpy array -> list
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj

# Helper functions for WSI reading
def _read_wsi_level_mir(wsi_path: str, level: int) -> np.ndarray:
    """
    Read an entire WSI level as a numpy array using multiresolutionimageinterface.

    Returns:
        np.ndarray with shape (H, W) for single-channel masks (or (H,W,C) if multi-channel).
    """
    import multiresolutionimageinterface as mir

    reader = mir.MultiResolutionImageReader()
    wsi = reader.open(str(wsi_path))

    if level < 0 or level >= wsi.getNumberOfLevels():
        raise ValueError(f"Invalid level {level}. Available levels: 0..{wsi.getNumberOfLevels()-1}")

    dims = wsi.getLevelDimensions(level)  # (W, H)
    w, h = int(dims[0]), int(dims[1])

    # read as uint8 if possible; MIR returns numpy array
    arr = wsi.getUCharPatch(0, 0, w, h, level)  # shape often (H,W) or (H,W,C)
    return arr


def _pick_level_for_metrics_mir(wsi_path: str, prefer: str = "lowest") -> int:
    """
    Pick a pyramid level for metric computation.
    - 'lowest' = smallest resolution (fastest, minimal memory)
    - 'highest' = full resolution (slowest, huge memory)
    """
    import multiresolutionimageinterface as mir
    reader = mir.MultiResolutionImageReader()
    wsi = reader.open(str(wsi_path))
    n_levels = wsi.getNumberOfLevels()

    if prefer == "highest":
        return 0
    if prefer == "lowest":
        return n_levels - 1

    # fallback
    return n_levels - 1


def _ensure_2d_mask(mask: np.ndarray) -> np.ndarray:
    """
    Ensure mask is 2D label image.
    If (H,W,3) etc, take first channel (common for stored masks).
    """
    if mask.ndim == 2:
        return mask
    if mask.ndim == 3:
        # common cases: (H,W,1) or (H,W,3)
        return mask[..., 0]
    raise ValueError(f"Unsupported mask shape: {mask.shape}")


class ArtifactMetricsWriter:
    """
    Computes quality metrics from artifact segmentation masks and saves them to YAML/JSON.
    """

    # Class name mapping
    CLASS_NAMES = {
        0: 'background',
        1: 'artifact_free',
        2: 'tissue_folds',
        3: 'ink',
        4: 'air_bubbles',
        5: 'dust',
        6: 'marker',
        7: 'out_of_focus'
    }

    # Classes that count as quality issues (exclude ink)
    QUALITY_ISSUE_CLASSES = [2, 4, 5, 6, 7]  # folds, air, dust, marker, focus

    def __init__(self, output_format: str = 'yaml'):
        """
        Initialize metrics writer.

        Args:
            output_format: Output format ('yaml' or 'json')
        """
        self.output_format = output_format.lower()
        if self.output_format not in ['yaml', 'json']:
            raise ValueError(f"output_format must be 'yaml' or 'json', got '{output_format}'")

    def compute_metrics(self, mask: np.ndarray, slide_id: str) -> Dict[str, Any]:
        """
        Compute quality metrics from a segmentation mask.

        Args:
            mask: Segmentation mask array with integer labels
            slide_id: Identifier for the slide

        Returns:
            Dictionary containing quality metrics
        """
        # Flatten mask for easier computation
        mask_flat = mask.flatten()

        # Count pixels per class
        unique, counts = np.unique(mask_flat, return_counts=True)
        class_counts = dict(zip(unique, counts))

        # Total pixels excluding background (class 0)
        total_pixels = np.sum([count for cls, count in class_counts.items() if cls != 0])

        if total_pixels == 0:
            return self._empty_metrics(slide_id, "No tissue detected (all background)")

        # Compute proportions for each artifact class
        proportions = {}
        for cls_id, cls_name in self.CLASS_NAMES.items():
            if cls_id == 0:  # Skip background
                continue
            pixel_count = class_counts.get(cls_id, 0)
            proportions[cls_name] = round(pixel_count / total_pixels, 6)

        # All artifact classes (everything except artifact_free class 1)
        artifact_classes = [cls for cls in self.CLASS_NAMES if cls not in (0, 1)]
        artifact_pixels = sum([class_counts.get(cls, 0) for cls in artifact_classes])
        artifact_proportion = round(artifact_pixels / total_pixels, 6)

        # Quality score including ink: 1 - artifact_proportion
        quality_score = round(1 - artifact_proportion, 6)

        # Quality score excluding ink (ink may be intentional/less clinically relevant)
        quality_issue_pixels = sum([class_counts.get(cls, 0) for cls in self.QUALITY_ISSUE_CLASSES])
        quality_score_excl_ink = round(1 - (quality_issue_pixels / total_pixels), 6)

        metrics = {
            'slide_id': slide_id,
            'artifact_proportion': artifact_proportion,        # All artifacts incl. ink (0-1; 0 = best)
            'quality_score': quality_score,                    # 1 - artifact_proportion (0-1; 1 = best)
            'quality_score_excl_ink': quality_score_excl_ink, # Ink not counted as artifact (0-1; 1 = best)
            'total_tissue_pixels': int(total_pixels),
            'artifact_proportions': {
                'tissue_folds': proportions['tissue_folds'],
                'ink': proportions['ink'],
                'air_bubbles': proportions['air_bubbles'],
                'dust': proportions['dust'],
                'marker': proportions['marker'],
                'out_of_focus': proportions['out_of_focus']
            }
        }

        return metrics

    def _empty_metrics(self, slide_id: str, reason: str) -> Dict[str, Any]:
        """Return empty metrics structure when computation fails."""
        return {
            'slide_id': slide_id,
            'error': reason,
            'artifact_proportion': None,
            'quality_score': None,
            'quality_score_excl_ink': None,
            'total_tissue_pixels': 0,
            'artifact_proportions': {}
        }

    def save_metrics(
        self,
        metrics: Dict[str, Any],
        output_path: str,
        append: bool = True
    ) -> None:
        """
        Save metrics to file, optionally appending to existing data.

        Args:
            metrics: Metrics dictionary for a single slide
            output_path: Path to output file
            append: If True, append to existing file; if False, overwrite
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing data if appending
        existing_data = {}
        if append and output_path.exists():
            try:
                with open(output_path, 'r') as f:
                    if self.output_format == 'yaml':
                        existing_data = yaml.safe_load(f) or {}
                    else:
                        existing_data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load existing metrics file: {e}")
                existing_data = {}

        # Update with new metrics (overwrites if slide_id already exists)
        slide_id = metrics['slide_id']
        existing_data[slide_id] = metrics

        # Save to file
        # Save to file
        existing_data = _to_builtin(existing_data)

        with open(output_path, 'w') as f:
            if self.output_format == 'yaml':
                yaml.safe_dump(existing_data, f, default_flow_style=False, sort_keys=False)
            else:
                json.dump(existing_data, f, indent=2, sort_keys=False)

        print(f"Metrics saved to: {output_path}")

    def load_metrics(self, input_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Load metrics from file.

        Args:
            input_path: Path to metrics file

        Returns:
            Dictionary mapping slide_ids to their metrics
        """
        input_path = Path(input_path)
        if not input_path.exists():
            return {}

        with open(input_path, 'r') as f:
            if self.output_format == 'yaml':
                return yaml.safe_load(f) or {}
            else:
                return json.load(f)

def compute_and_save_metrics(
    mask_path: str,
    slide_id: str,
    output_path: str,
    output_format: str = 'yaml',
    append: bool = True,
    wsi_level_preference: str = "lowest",  # "lowest" or "highest"
) -> Dict[str, Any]:
    """
    Convenience function to compute and save metrics from a mask file.
    Supports WSI pyramids via multiresolutionimageinterface.
    """
    writer = ArtifactMetricsWriter(output_format=output_format)

    mask_path = str(mask_path)
    try:
        # If it looks like a WSI / pyramidal mask, read with MIR.
        # (MIR can open many formats; this is fine even for non-pyramidal TIFFs.)
        level = _pick_level_for_metrics_mir(mask_path, prefer=wsi_level_preference)
        mask = _read_wsi_level_mir(mask_path, level=level)
        mask = _ensure_2d_mask(mask).astype(np.uint8)

    except Exception as e:
        # fallback: old behavior (handy if MIR isn’t available in some env)
        try:
            from PIL import Image
            mask = np.array(Image.open(mask_path))
            mask = _ensure_2d_mask(mask).astype(np.uint8)
        except Exception as e2:
            metrics = writer._empty_metrics(slide_id, f"Failed to load mask with MIR/PIL: {e} / {e2}")
            writer.save_metrics(metrics, output_path, append=append)
            return metrics

    metrics = writer.compute_metrics(mask, slide_id)
    writer.save_metrics(metrics, output_path, append=append)
    return metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute quality metrics from artifact segmentation masks'
    )
    parser.add_argument('--mask_path', type=str, required=True,
                        help='Path to segmentation mask file')
    parser.add_argument('--slide_id', type=str, required=True,
                        help='Slide identifier')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to output metrics file')
    parser.add_argument('--format', type=str, default='yaml', choices=['yaml', 'json'],
                        help='Output format (yaml or json)')
    parser.add_argument('--no-append', action='store_true',
                        help='Overwrite instead of appending to existing file')
    parser.add_argument(
        '--wsi_level_preference',
        type=str,
        default='lowest',
        choices=['lowest', 'highest'],
        help="Which WSI pyramid level to use for metrics (default: lowest = fastest)."
    )


    args = parser.parse_args()

    metrics = compute_and_save_metrics(
        mask_path=args.mask_path,
        slide_id=args.slide_id,
        output_path=args.output_path,
        output_format=args.format,
        append=not args.no_append
    )

    print(f"\nArtifact proportion: {metrics['artifact_proportion']:.4f}")
    print(f"Quality Score: {metrics['quality_score']:.4f}")
    print(f"Quality Score (excl. ink): {metrics['quality_score_excl_ink']:.4f}")
    if 'artifact_proportions' in metrics and 'ink' in metrics['artifact_proportions']:
        print(f"Ink: {metrics['artifact_proportions']['ink']:.4f}")
