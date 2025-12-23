import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union
from pathlib import Path

logger = logging.getLogger(__name__)

# Type for custom duration extractor function
DurationExtractor = Callable[[str, float, int], List[float]]


def compute_adaptive_ranges(
    json_file: str,
    num_levels: int = 6,
    scale_factor: int = 2,
    fps: float = 30.0,
    feat_stride: int = 1,
    overlap_ratio: float = 0.5,
    min_range: float = 0.0,
    duration_extractor: Optional[DurationExtractor] = None,
    durations: Optional[List[float]] = None,
) -> List[Tuple[int, int]]:
    """
    Compute adaptive regression ranges based on event duration distribution.

    Args:
        json_file: Path to annotation JSON file (ignored if durations provided)
        num_levels: Number of FPN levels
        scale_factor: Downsampling factor between levels
        fps: Video FPS
        feat_stride: Feature stride (frames per feature)
        overlap_ratio: Overlap between adjacent level ranges
        min_range: Minimum range value for first level
        duration_extractor: Custom function to extract durations from json_file.
                           Signature: (json_file, fps, feat_stride) -> List[float]
                           If None, uses ActivityNet format extractor.
        durations: Pre-computed durations list (skips file loading if provided)

    Returns:
        List of (min, max) tuples for each FPN level
    """
    if durations is not None:
        extracted = durations
    elif duration_extractor is not None:
        extracted = duration_extractor(json_file, fps, feat_stride)
    else:
        extracted = _extract_durations_activitynet(json_file, fps, feat_stride)

    if len(extracted) == 0:
        logger.warning("No annotations found, using default ranges")
        return _default_ranges(num_levels, scale_factor)

    durations_arr = np.array(extracted)
    logger.info(f"Analyzing {len(durations_arr)} events")
    logger.info(f"Duration stats: min={durations_arr.min():.1f}, max={durations_arr.max():.1f}, "
                f"median={np.median(durations_arr):.1f}, mean={durations_arr.mean():.1f}")

    ranges = _compute_quantile_ranges(durations_arr, num_levels, overlap_ratio, min_range)

    _validate_ranges(ranges, durations_arr)

    return ranges


def _extract_durations_activitynet(json_file: str, fps: float, feat_stride: int) -> List[float]:
    """Extract event durations from ActivityNet-style annotation file."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    durations = []
    database = data.get('database', {})

    for video_id, video_info in database.items():
        annotations = video_info.get('annotations', [])
        for ann in annotations:
            segment = ann.get('segment', [])
            if len(segment) == 2:
                start, end = segment
                duration_sec = end - start
                duration_feats = (duration_sec * fps) / feat_stride
                if duration_feats > 0:
                    durations.append(duration_feats)

    return durations


def _compute_quantile_ranges(
    durations: np.ndarray,
    num_levels: int,
    overlap_ratio: float,
    min_range: float,
) -> List[Tuple[int, int]]:
    """Compute ranges using quantile-based boundaries."""
    quantiles = np.linspace(0, 1, num_levels + 1)
    boundaries = np.percentile(durations, quantiles * 100)

    boundaries[0] = max(min_range, 0)
    boundaries[-1] = max(boundaries[-1], durations.max() * 1.1)

    ranges = []
    for i in range(num_levels):
        if i == 0:
            range_min = int(boundaries[i])
        else:
            overlap = (boundaries[i] - boundaries[i-1]) * overlap_ratio
            range_min = int(max(0, boundaries[i] - overlap))

        if i == num_levels - 1:
            range_max = int(np.ceil(boundaries[i+1]))
        else:
            overlap = (boundaries[i+2] - boundaries[i+1]) * overlap_ratio if i+2 <= num_levels else 0
            range_max = int(np.ceil(boundaries[i+1] + overlap))

        ranges.append((range_min, range_max))

    return ranges


def _default_ranges(num_levels: int, scale_factor: int) -> List[Tuple[int, int]]:
    """Generate default exponential ranges."""
    ranges = []
    for i in range(num_levels):
        range_min = 0 if i == 0 else scale_factor ** (i - 1)
        range_max = scale_factor ** i if i < num_levels - 1 else scale_factor ** i * 2
        ranges.append((range_min, range_max))
    return ranges


def _validate_ranges(ranges: List[Tuple[int, int]], durations: np.ndarray) -> None:
    """Validate ranges cover event distribution."""
    coverage = np.zeros(len(durations), dtype=bool)

    for i, (rmin, rmax) in enumerate(ranges):
        in_range = (durations >= rmin) & (durations <= rmax)
        coverage |= in_range
        pct = in_range.sum() / len(durations) * 100
        logger.info(f"Level {i}: range=({rmin}, {rmax}), coverage={pct:.1f}%")

    total_coverage = coverage.sum() / len(durations) * 100
    logger.info(f"Total coverage: {total_coverage:.1f}%")

    if total_coverage < 95:
        logger.warning(f"Low coverage ({total_coverage:.1f}%). Consider adjusting ranges.")


def suggest_config_ranges(
    json_file: str,
    backbone_arch: List[int],
    fpn_start_level: int = 0,
    **kwargs
) -> Dict:
    """
    Suggest complete config settings for regression ranges.

    Args:
        json_file: Path to annotation JSON
        backbone_arch: Model backbone architecture [start, ..., end]
        fpn_start_level: Starting FPN level
        **kwargs: Additional args for compute_adaptive_ranges

    Returns:
        Dict with suggested config values
    """
    num_levels = backbone_arch[-1] + 1 - fpn_start_level
    ranges = compute_adaptive_ranges(json_file, num_levels=num_levels, **kwargs)

    return {
        'model': {
            'backbone_arch': backbone_arch,
            'fpn_start_level': fpn_start_level,
            'regression_range': ranges,
        },
        '_metadata': {
            'source_file': str(json_file),
            'num_levels': num_levels,
        }
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compute adaptive regression ranges')
    parser.add_argument('json_file', help='Path to annotation JSON')
    parser.add_argument('--num-levels', type=int, default=6)
    parser.add_argument('--scale-factor', type=int, default=2)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--feat-stride', type=int, default=1)
    parser.add_argument('--overlap', type=float, default=0.5)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    ranges = compute_adaptive_ranges(
        args.json_file,
        num_levels=args.num_levels,
        scale_factor=args.scale_factor,
        fps=args.fps,
        feat_stride=args.feat_stride,
        overlap_ratio=args.overlap,
    )

    print("\nSuggested regression_range:")
    print(f"  regression_range: {ranges}")
