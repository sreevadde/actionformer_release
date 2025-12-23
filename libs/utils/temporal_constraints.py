"""
Temporal constraint enforcement for action detection results.

Useful for domains where actions follow specific temporal patterns,
such as sports events (plays follow each other), procedures, etc.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


class TemporalConstraints:
    """
    Enforces temporal constraints on detection results.

    Constraints:
    - min_gap: Minimum time gap between events of same class
    - max_overlap: Maximum allowed overlap ratio between events
    - duration_range: (min, max) duration limits per class
    - ordering: Enforce specific class orderings

    Usage:
        constraints = TemporalConstraints(
            min_gap={0: 1.0},  # Class 0 events must be 1s apart
            duration_range={0: (0.1, 2.0)}  # Class 0 duration limits
        )
        filtered = constraints.apply(results)
    """

    def __init__(
        self,
        min_gap: Optional[Dict[int, float]] = None,
        max_overlap: float = 0.5,
        duration_range: Optional[Dict[int, Tuple[float, float]]] = None,
        ordering: Optional[List[List[int]]] = None,
        score_threshold: float = 0.0,
    ):
        self.min_gap = min_gap or {}
        self.max_overlap = max_overlap
        self.duration_range = duration_range or {}
        self.ordering = ordering
        self.score_threshold = score_threshold

    def apply(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply constraints to results dict.

        Args:
            results: Dict with video-id, t-start, t-end, label, score arrays

        Returns:
            Filtered results dict
        """
        if len(results.get('video-id', [])) == 0:
            return results

        video_ids = np.array(results['video-id'])
        t_start = np.array(results['t-start'])
        t_end = np.array(results['t-end'])
        labels = np.array(results['label'])
        scores = np.array(results['score'])

        keep_mask = np.ones(len(video_ids), dtype=bool)
        keep_mask &= self._filter_duration(t_start, t_end, labels)
        keep_mask &= self._filter_min_gap(video_ids, t_start, t_end, labels, scores)
        keep_mask &= self._filter_overlap(video_ids, t_start, t_end, labels, scores)
        keep_mask &= scores >= self.score_threshold

        return {
            'video-id': video_ids[keep_mask].tolist(),
            't-start': t_start[keep_mask],
            't-end': t_end[keep_mask],
            'label': labels[keep_mask],
            'score': scores[keep_mask],
        }

    def _filter_duration(
        self,
        t_start: np.ndarray,
        t_end: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """Filter events outside duration bounds."""
        if not self.duration_range:
            return np.ones(len(labels), dtype=bool)

        durations = t_end - t_start
        keep = np.ones(len(labels), dtype=bool)

        for cls_id, (min_dur, max_dur) in self.duration_range.items():
            cls_mask = labels == cls_id
            valid_dur = (durations >= min_dur) & (durations <= max_dur)
            keep &= ~cls_mask | valid_dur

        return keep

    def _filter_min_gap(
        self,
        video_ids: np.ndarray,
        t_start: np.ndarray,
        t_end: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray
    ) -> np.ndarray:
        """Filter events that are too close to higher-scoring events."""
        if not self.min_gap:
            return np.ones(len(labels), dtype=bool)

        keep = np.ones(len(labels), dtype=bool)

        for video_id in np.unique(video_ids):
            vid_mask = video_ids == video_id

            for cls_id, gap in self.min_gap.items():
                cls_vid_mask = vid_mask & (labels == cls_id)
                indices = np.where(cls_vid_mask)[0]

                if len(indices) <= 1:
                    continue

                idx_scores = scores[indices]
                sorted_order = np.argsort(-idx_scores)
                sorted_indices = indices[sorted_order]

                kept_events = []
                for idx in sorted_indices:
                    event_start, event_end = t_start[idx], t_end[idx]
                    event_center = (event_start + event_end) / 2

                    too_close = False
                    for kept_idx in kept_events:
                        kept_center = (t_start[kept_idx] + t_end[kept_idx]) / 2
                        if abs(event_center - kept_center) < gap:
                            too_close = True
                            break

                    if too_close:
                        keep[idx] = False
                    else:
                        kept_events.append(idx)

        return keep

    def _filter_overlap(
        self,
        video_ids: np.ndarray,
        t_start: np.ndarray,
        t_end: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray
    ) -> np.ndarray:
        """Filter events with high overlap (keeping higher scoring one)."""
        if self.max_overlap >= 1.0:
            return np.ones(len(labels), dtype=bool)

        keep = np.ones(len(labels), dtype=bool)

        for video_id in np.unique(video_ids):
            vid_mask = video_ids == video_id
            indices = np.where(vid_mask)[0]

            if len(indices) <= 1:
                continue

            sorted_order = np.argsort(-scores[indices])
            sorted_indices = indices[sorted_order]

            for i, idx in enumerate(sorted_indices):
                if not keep[idx]:
                    continue

                for j in range(i + 1, len(sorted_indices)):
                    other_idx = sorted_indices[j]
                    if not keep[other_idx]:
                        continue

                    overlap = self._compute_overlap(
                        t_start[idx], t_end[idx],
                        t_start[other_idx], t_end[other_idx]
                    )

                    if overlap > self.max_overlap:
                        keep[other_idx] = False

        return keep

    def _compute_overlap(
        self,
        start1: float, end1: float,
        start2: float, end2: float
    ) -> float:
        """Compute IoU-style overlap between two segments."""
        inter_start = max(start1, start2)
        inter_end = min(end1, end2)
        intersection = max(0, inter_end - inter_start)

        dur1 = end1 - start1
        dur2 = end2 - start2
        union = dur1 + dur2 - intersection

        return intersection / union if union > 0 else 0.0


def enforce_class_constraints(
    results: Dict[str, Any],
    class_id: int,
    min_gap: Optional[float] = None,
    duration_range: Optional[Tuple[float, float]] = None,
    max_overlap: float = 0.3,
) -> Dict[str, Any]:
    """
    Apply temporal constraints to a specific class.

    Args:
        results: Detection results dict
        class_id: Class ID to constrain
        min_gap: Minimum time gap between events (seconds)
        duration_range: (min, max) duration limits (seconds)
        max_overlap: Maximum IoU overlap

    Returns:
        Filtered results
    """
    constraints = TemporalConstraints(
        min_gap={class_id: min_gap} if min_gap else None,
        duration_range={class_id: duration_range} if duration_range else None,
        max_overlap=max_overlap,
    )
    return constraints.apply(results)


def merge_close_detections(
    results: Dict[str, Any],
    merge_gap: float = 0.5,
    class_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Merge detections that are very close together.

    Args:
        results: Detection results dict
        merge_gap: Maximum gap to merge (seconds)
        class_id: Only merge specific class (None = merge all classes separately)

    Returns:
        Results with close detections merged
    """
    if len(results.get('video-id', [])) == 0:
        return results

    video_ids = np.array(results['video-id'])
    t_start = np.array(results['t-start'])
    t_end = np.array(results['t-end'])
    labels = np.array(results['label'])
    scores = np.array(results['score'])

    new_results = {
        'video-id': [],
        't-start': [],
        't-end': [],
        'label': [],
        'score': [],
    }

    processed = np.zeros(len(video_ids), dtype=bool)

    for video_id in np.unique(video_ids):
        vid_mask = video_ids == video_id
        classes = [class_id] if class_id is not None else np.unique(labels[vid_mask])

        for cls in classes:
            cls_mask = vid_mask & (labels == cls)
            indices = np.where(cls_mask)[0]

            if len(indices) == 0:
                continue

            sorted_order = np.argsort(t_start[indices])
            sorted_indices = indices[sorted_order]
            processed[sorted_indices] = True

            i = 0
            while i < len(sorted_indices):
                group = [sorted_indices[i]]
                j = i + 1

                while j < len(sorted_indices):
                    prev_end = t_end[group[-1]]
                    curr_start = t_start[sorted_indices[j]]

                    if curr_start - prev_end <= merge_gap:
                        group.append(sorted_indices[j])
                        j += 1
                    else:
                        break

                new_results['video-id'].append(video_id)
                new_results['t-start'].append(t_start[group[0]])
                new_results['t-end'].append(max(t_end[idx] for idx in group))
                new_results['label'].append(cls)
                new_results['score'].append(max(scores[idx] for idx in group))

                i = j

    unprocessed = np.where(~processed)[0]
    for idx in unprocessed:
        new_results['video-id'].append(video_ids[idx])
        new_results['t-start'].append(t_start[idx])
        new_results['t-end'].append(t_end[idx])
        new_results['label'].append(labels[idx])
        new_results['score'].append(scores[idx])

    return {
        'video-id': new_results['video-id'],
        't-start': np.array(new_results['t-start']) if new_results['t-start'] else np.array([]),
        't-end': np.array(new_results['t-end']) if new_results['t-end'] else np.array([]),
        'label': np.array(new_results['label']) if new_results['label'] else np.array([]),
        'score': np.array(new_results['score']) if new_results['score'] else np.array([]),
    }
