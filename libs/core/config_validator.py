import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigValidator:
    """
    Validates ActionFormer configuration to catch errors early.

    Usage:
        validator = ConfigValidator(cfg)
        validator.validate()  # Raises ConfigurationError if invalid

        # Less strict mode (warnings only for soft constraints)
        validator = ConfigValidator(cfg, strict=False)
    """

    def __init__(self, cfg: Dict[str, Any], strict: bool = True):
        self.cfg = cfg
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def _add_issue(self, msg: str, is_hard_error: bool = True) -> None:
        """Add error or warning based on strict mode."""
        if is_hard_error:
            self.errors.append(msg)
        elif self.strict:
            self.errors.append(msg)
        else:
            self.warnings.append(msg)

    def validate(self) -> None:
        """Run all validation checks. Raises ConfigurationError if invalid."""
        self._check_required_fields()
        self._check_fpn_regression_alignment()
        self._check_backbone_params()
        self._check_sequence_divisibility()
        self._check_head_params()
        self._check_loss_params()
        self._check_nms_params()

        if self.errors:
            error_msg = "Configuration errors:\n" + "\n".join(f"  - {e}" for e in self.errors)
            raise ConfigurationError(error_msg)

        if self.warnings:
            for w in self.warnings:
                logger.warning(f"Config: {w}")

    def _check_required_fields(self) -> None:
        """Check that all required fields are present."""
        required = [
            ('dataset', 'json_file'),
            ('dataset', 'feat_folder'),
            ('dataset', 'max_seq_len'),
            ('model', 'backbone_type'),
            ('model', 'backbone_arch'),
            ('model', 'regression_range'),
        ]

        for section, field in required:
            if section not in self.cfg:
                self.errors.append(f"Missing required section: {section}")
            elif field not in self.cfg[section]:
                self.errors.append(f"Missing required field: {section}.{field}")

    def _check_fpn_regression_alignment(self) -> None:
        """Check FPN levels match regression ranges."""
        if 'model' not in self.cfg:
            return

        model_cfg = self.cfg['model']

        if 'backbone_arch' not in model_cfg or 'regression_range' not in model_cfg:
            return

        arch = model_cfg['backbone_arch']
        fpn_start = model_cfg.get('fpn_start_level', 0)

        n_levels = arch[-1] + 1 - fpn_start
        n_ranges = len(model_cfg['regression_range'])

        if n_levels != n_ranges:
            self.errors.append(
                f"FPN levels ({n_levels}) must equal regression ranges ({n_ranges}). "
                f"backbone_arch={arch}, fpn_start_level={fpn_start}"
            )

        ranges = model_cfg['regression_range']
        for i, r in enumerate(ranges):
            if len(r) != 2:
                self.errors.append(f"regression_range[{i}] must have 2 elements, got {len(r)}")
            elif r[0] >= r[1]:
                self.errors.append(f"regression_range[{i}]: min ({r[0]}) must be < max ({r[1]})")

    def _check_backbone_params(self) -> None:
        """Check backbone-specific parameters."""
        if 'model' not in self.cfg:
            return

        model_cfg = self.cfg['model']
        backbone_type = model_cfg.get('backbone_type', '')
        backbone_cfg = model_cfg.get('backbone', {})

        if backbone_type == 'convTransformerv2':
            if backbone_cfg.get('use_rel_pe', False):
                self.warnings.append(
                    "convTransformerv2 ignores use_rel_pe. Use use_rope instead."
                )

            if backbone_cfg.get('use_rope', False) and backbone_cfg.get('use_abs_pe', True):
                self.warnings.append(
                    "Using both RoPE and absolute PE. Consider disabling use_abs_pe."
                )

        elif backbone_type == 'convTransformer':
            v2_params = ['use_rope', 'use_flash_attn', 'use_swiglu', 'use_rms_norm']
            for param in v2_params:
                if backbone_cfg.get(param, False):
                    self.warnings.append(
                        f"{param} is ignored for convTransformer. Use convTransformerv2."
                    )

    def _check_sequence_divisibility(self) -> None:
        """Check sequence length is divisible by required strides."""
        if 'dataset' not in self.cfg or 'model' not in self.cfg:
            return

        max_seq = self.cfg['dataset'].get('max_seq_len', 2304)
        model_cfg = self.cfg['model']

        arch = model_cfg.get('backbone_arch', [2, 2, 5])
        scale_factor = model_cfg.get('scale_factor', 2)
        win_size = model_cfg.get('n_mha_win_size', 19)

        if isinstance(win_size, int):
            win_sizes = [win_size] * (1 + arch[-1])
        else:
            win_sizes = win_size

        fpn_start = model_cfg.get('fpn_start_level', 0)

        for level in range(fpn_start, arch[-1] + 1):
            w = win_sizes[level] if level < len(win_sizes) else win_sizes[-1]
            stride = (scale_factor ** level)
            if w > 1:
                stride = stride * (w // 2) * 2

            if max_seq % stride != 0:
                self.errors.append(
                    f"max_seq_len ({max_seq}) must be divisible by stride ({stride}) "
                    f"at FPN level {level}. Consider max_seq_len={max_seq - (max_seq % stride)}"
                )

    def _check_head_params(self) -> None:
        """Check detection head parameters."""
        if 'model' not in self.cfg:
            return

        model_cfg = self.cfg['model']

        head_layers = model_cfg.get('head_num_layers', 3)
        if head_layers < 1:
            self.errors.append(f"head_num_layers must be >= 1, got {head_layers}")
        elif head_layers < 2:
            self.warnings.append(f"head_num_layers={head_layers} may limit model capacity")

        head_dim = model_cfg.get('head_dim', 256)
        fpn_dim = model_cfg.get('fpn_dim', 256)
        if head_dim != fpn_dim:
            self.warnings.append(
                f"head_dim ({head_dim}) != fpn_dim ({fpn_dim}). This adds projection layer."
            )

    def _check_loss_params(self) -> None:
        """Check loss configuration."""
        train_cfg = self.cfg.get('train_cfg', {})

        loss_weight = train_cfg.get('loss_weight', 1.0)
        if loss_weight < 0:
            self.errors.append(f"loss_weight must be >= 0, got {loss_weight}")

        label_smoothing = train_cfg.get('label_smoothing', 0.0)
        if not 0 <= label_smoothing < 1:
            self.errors.append(f"label_smoothing must be in [0, 1), got {label_smoothing}")

        reg_loss_type = train_cfg.get('reg_loss_type', 'diou')
        known_reg_losses = ['diou', 'eiou', 'giou', 'iou']
        if reg_loss_type not in known_reg_losses:
            self.warnings.append(
                f"reg_loss_type '{reg_loss_type}' not in known types {known_reg_losses}"
            )

    def _check_nms_params(self) -> None:
        """Check NMS configuration."""
        test_cfg = self.cfg.get('test_cfg', {})

        nms_method = test_cfg.get('nms_method', 'soft')
        known_methods = ['soft', 'hard', 'none']
        if nms_method not in known_methods:
            self.warnings.append(
                f"nms_method '{nms_method}' not in known methods {known_methods}"
            )

        iou_thresh = test_cfg.get('iou_threshold', 0.1)
        if not 0 < iou_thresh <= 1:
            self.errors.append(f"iou_threshold must be in (0, 1], got {iou_thresh}")

        nms_sigma = test_cfg.get('nms_sigma', 0.5)
        if nms_sigma <= 0:
            self.errors.append(f"nms_sigma must be > 0, got {nms_sigma}")

        temp = test_cfg.get('score_temperature', 1.0)
        if temp <= 0:
            self.errors.append(f"score_temperature must be > 0, got {temp}")


def validate_config(cfg: Dict[str, Any], strict: bool = True) -> None:
    """
    Validate configuration dictionary.

    Args:
        cfg: Configuration dictionary
        strict: If True, treat soft constraints as errors. If False, only warn.

    Raises:
        ConfigurationError: If configuration is invalid
    """
    validator = ConfigValidator(cfg, strict=strict)
    validator.validate()
