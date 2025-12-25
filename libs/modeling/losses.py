import torch
from torch.nn import functional as F

@torch.jit.script
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Taken from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = 0.25.
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


@torch.jit.script
def ctr_giou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Generalized Intersection over Union Loss (Hamid Rezatofighi et. al)
    https://arxiv.org/abs/1902.09630

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # giou is reduced to iou in our setting, skip unnecessary steps
    loss = 1.0 - iouk

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

@torch.jit.script
def ctr_diou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Distance-IoU Loss (Zheng et. al)
    https://arxiv.org/abs/1911.08287

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # smallest enclosing box
    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)
    len_c = lc + rc

    # offset between centers
    rho = 0.5 * (rp - lp - rg + lg)

    # diou
    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


@torch.jit.script
def ctr_eiou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Enhanced IoU Loss (EIoU) for 1D temporal segments.
    Extends DIoU with aspect ratio penalty for better convergence.

    Reference: https://arxiv.org/abs/2101.08158

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)
    intsctk = rkis + lkis

    # union
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # enclosing box
    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)
    len_c = lc + rc

    # center distance penalty (DIoU term)
    rho = 0.5 * (rp - lp - rg + lg)
    d2 = torch.square(rho)
    c2 = torch.square(len_c).clamp(min=eps)

    # aspect ratio penalty (EIoU term) - penalize duration differences
    len_p = lp + rp
    len_g = lg + rg
    rho_w = torch.square(len_p - len_g)

    # EIoU = IoU - d²/c² - ρ_w²/c²
    loss = 1.0 - iouk + d2 / c2 + rho_w / c2

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


@torch.jit.script
def quality_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    iou_targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Quality Focal Loss - combines focal loss with IoU-aware quality signal.
    Aligns classification confidence with localization quality.

    Reference: https://arxiv.org/abs/2006.04388

    Args:
        inputs: Classification logits (N, C)
        targets: Binary classification targets (N, C)
        iou_targets: IoU quality scores for positive samples (N,)
        alpha: Focal loss alpha parameter
        gamma: Focal loss gamma parameter
        reduction: 'none' | 'mean' | 'sum'
    """
    inputs = inputs.float()
    targets = targets.float()
    iou_targets = iou_targets.float()

    p = torch.sigmoid(inputs)

    # For positive samples, target becomes IoU quality
    # For negative samples, target remains 0
    quality_targets = targets * iou_targets.unsqueeze(-1).clamp(min=0.0, max=1.0)

    # BCE with quality targets
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, quality_targets, reduction="none"
    )

    # Focal weight based on distance from quality target
    focal_weight = torch.abs(quality_targets - p) ** gamma

    # Alpha weighting
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        focal_weight = alpha_t * focal_weight

    loss = focal_weight * ce_loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


@torch.jit.script
def focal_regression_loss(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    gamma: float = 2.0,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Focal Regression Loss - focuses on harder regression samples.
    Downweights easy samples with high IoU, focuses on boundary cases.

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        gamma: Focusing parameter (higher = more focus on hard samples)
        reduction: 'none' | 'mean' | 'sum'
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)
    intsctk = rkis + lkis

    # union
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # base loss (1 - IoU)
    base_loss = 1.0 - iouk

    # focal weight: focus on hard samples (low IoU)
    focal_weight = (1.0 - iouk) ** gamma

    loss = focal_weight * base_loss

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


@torch.jit.script
def iou_weighted_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    base_loss_type: str = 'diou',
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    IoU-Weighted Regression Loss - softer version of focal regression.
    Weights loss by (1 - IoU) linearly, not exponentially.

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        base_loss_type: 'iou' | 'diou' | 'eiou' - which base loss to use
        reduction: 'none' | 'mean' | 'sum'
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)
    intsctk = rkis + lkis

    # union
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # base loss is 1 - IoU
    base_loss = 1.0 - iouk

    if base_loss_type == 'diou' or base_loss_type == 'eiou':
        # add center distance penalty
        lc = torch.max(lp, lg)
        rc = torch.max(rp, rg)
        len_c = lc + rc
        rho = 0.5 * (rp - lp - rg + lg)
        d2 = torch.square(rho)
        c2 = torch.square(len_c).clamp(min=eps)
        base_loss = base_loss + d2 / c2

        if base_loss_type == 'eiou':
            # add width penalty
            len_p = lp + rp
            len_g = lg + rg
            rho_w = torch.square(len_p - len_g)
            base_loss = base_loss + rho_w / c2

    weight = 2.0 - iouk

    loss = weight * base_loss

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def compute_iou_1d(
    pred_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute IoU for 1D segments represented as center offsets.

    Args:
        pred_offsets: Predicted offsets (N, 2)
        target_offsets: Target offsets (N, 2)

    Returns:
        IoU scores (N,)
    """
    lp, rp = pred_offsets[:, 0], pred_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)
    intsctk = (rkis + lkis).clamp(min=0)

    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    return iouk


@torch.jit.script
def gaussian_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 2.0,
    gamma: float = 4.0,
) -> torch.Tensor:
    """
    Gaussian Focal Loss for heatmap regression (CornerNet style).

    Args:
        inputs: Logits before sigmoid
        targets: Gaussian heatmap targets (peaks = 1.0)
        alpha: Focal modulating factor
        gamma: Negative sample weighting factor
    """
    pred = torch.sigmoid(inputs)
    pos_inds = targets.eq(1)
    neg_inds = targets.lt(1)

    neg_weights = torch.pow(1 - targets, gamma)
    pos_loss = torch.log(pred.clamp(min=1e-12)) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log((1 - pred).clamp(min=1e-12)) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos

    return loss


@torch.jit.script
def distribution_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reg_max: int = 16,
) -> torch.Tensor:
    """
    Distribution Focal Loss (DFL) from Generalized Focal Loss.

    Learns discrete probability distribution over regression bins.
    Final prediction is expectation over the distribution.

    Args:
        pred: Predicted distribution logits, shape (N, reg_max + 1)
        target: Continuous regression target, shape (N,), values in [0, reg_max]
        reg_max: Maximum regression range (number of bins - 1)
    """
    target = target.clamp(0, reg_max - 0.01)
    left = target.long()
    right = left + 1
    weight_left = right.float() - target
    weight_right = target - left.float()

    loss = (
        F.cross_entropy(pred, left, reduction='none') * weight_left +
        F.cross_entropy(pred, right, reduction='none') * weight_right
    )
    return loss


def decode_distribution(pred: torch.Tensor, reg_max: int = 16) -> torch.Tensor:
    """
    Decode distribution prediction to scalar offset.

    Args:
        pred: Distribution logits, shape (..., reg_max + 1)
        reg_max: Maximum regression range

    Returns:
        Scalar offset as expectation over distribution
    """
    bins = torch.arange(reg_max + 1, device=pred.device, dtype=pred.dtype)
    probs = F.softmax(pred, dim=-1)
    return (probs * bins).sum(dim=-1)
