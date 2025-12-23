import logging
from collections import defaultdict
from typing import Dict, List, Optional
import torch

logger = logging.getLogger(__name__)


class LossLogger:
    """
    Enhanced loss logging with moving averages and detailed breakdowns.

    Usage:
        loss_logger = LossLogger(window_size=100)

        # In training loop:
        loss_logger.update(loss_dict, extra_info={'num_pos': num_pos})

        # Periodic logging:
        if step % log_interval == 0:
            loss_logger.log_summary(step)
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.step_count = 0

    def update(
        self,
        losses: Dict[str, torch.Tensor],
        extra_info: Optional[Dict[str, float]] = None
    ) -> None:
        """Record losses for current step."""
        self.step_count += 1

        for name, value in losses.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().item()
            self.history[name].append(value)
            if len(self.history[name]) > self.window_size:
                self.history[name].pop(0)

        if extra_info:
            for name, value in extra_info.items():
                self.history[name].append(value)
                if len(self.history[name]) > self.window_size:
                    self.history[name].pop(0)

    def get_averages(self) -> Dict[str, float]:
        """Get moving averages for all tracked values."""
        return {
            name: sum(values) / len(values)
            for name, values in self.history.items()
            if len(values) > 0
        }

    def log_summary(self, step: int, prefix: str = "") -> str:
        """Log summary and return formatted string."""
        avgs = self.get_averages()
        if not avgs:
            return ""

        parts = [f"Step {step}:"]

        core_losses = ['final_loss', 'cls_loss', 'reg_loss']
        for name in core_losses:
            if name in avgs:
                parts.append(f"{name}={avgs[name]:.4f}")

        extra = {k: v for k, v in avgs.items() if k not in core_losses}
        for name, value in sorted(extra.items()):
            if name in ['num_pos', 'num_neg']:
                parts.append(f"{name}={value:.0f}")
            elif name in ['mean_iou', 'mean_score']:
                parts.append(f"{name}={value:.3f}")
            else:
                parts.append(f"{name}={value:.4f}")

        msg = f"{prefix}{' | '.join(parts)}"
        logger.info(msg)
        return msg

    def get_tensorboard_dict(self, prefix: str = "train/") -> Dict[str, float]:
        """Get dict formatted for tensorboard logging."""
        return {
            f"{prefix}{name}": value
            for name, value in self.get_averages().items()
        }

    def reset(self) -> None:
        """Clear all history."""
        self.history.clear()
        self.step_count = 0


def compute_loss_breakdown(
    cls_logits_list: List[torch.Tensor],
    offsets_list: List[torch.Tensor],
    gt_cls_labels: List[torch.Tensor],
    gt_offsets: List[torch.Tensor],
    fpn_masks: List[torch.Tensor],
) -> Dict[str, float]:
    """
    Compute per-level loss breakdown for debugging.

    Returns dict with per-level positive counts and rough loss estimates.
    """
    breakdown = {}

    gt_cls = torch.stack(gt_cls_labels)

    level_offset = 0
    for level_idx, mask in enumerate(fpn_masks):
        level_len = mask.shape[1]

        level_gt = gt_cls[:, level_offset:level_offset + level_len]
        level_mask = mask

        level_pos = (level_gt.sum(-1) > 0) & level_mask
        num_pos = level_pos.sum().item()

        breakdown[f'level_{level_idx}_pos'] = num_pos
        breakdown[f'level_{level_idx}_total'] = level_mask.sum().item()

        level_offset += level_len

    return breakdown


def compute_iou_stats(
    pred_offsets: torch.Tensor,
    gt_offsets: torch.Tensor,
    eps: float = 1e-8
) -> Dict[str, float]:
    """Compute IoU statistics for positive samples."""
    if pred_offsets.shape[0] == 0:
        return {'mean_iou': 0.0, 'min_iou': 0.0, 'max_iou': 0.0}

    lp, rp = pred_offsets[:, 0], pred_offsets[:, 1]
    lg, rg = gt_offsets[:, 0], gt_offsets[:, 1]

    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)
    intsct = (rkis + lkis).clamp(min=0)

    union = (lp + rp) + (lg + rg) - intsct
    iou = intsct / union.clamp(min=eps)

    return {
        'mean_iou': iou.mean().item(),
        'min_iou': iou.min().item(),
        'max_iou': iou.max().item(),
    }
