# Framework Improvements

Utilities for configuration validation, regression range computation, and post-processing.

## Configuration Validation

```python
from libs.core import validate_config, ConfigurationError

cfg = load_config(args.config)
try:
    validate_config(cfg)  # strict=True by default
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    sys.exit(1)

# Less strict mode (soft constraints become warnings)
validate_config(cfg, strict=False)
```

### Checks Performed

| Check | Type | Description |
|-------|------|-------------|
| Required fields | Error | dataset.json_file, model.backbone_type, etc. |
| FPN/range alignment | Error | regression_range count matches FPN levels |
| Sequence divisibility | Error | max_seq_len divisible by all strides |
| head_num_layers < 1 | Error | Must have at least 1 layer |
| head_num_layers < 2 | Warning | May limit capacity |
| Unknown loss type | Warning | Not in known types |
| Unknown NMS method | Warning | Not in known methods |

## Adaptive Regression Ranges

Compute regression ranges from annotation duration distribution.

```python
from libs.core import compute_adaptive_ranges

# From ActivityNet-format JSON
ranges = compute_adaptive_ranges(
    json_file='data/annotations.json',
    num_levels=6,
    fps=30.0,
    feat_stride=4,
)

# From pre-computed durations (any format)
ranges = compute_adaptive_ranges(
    json_file='',
    num_levels=6,
    durations=[10.5, 25.0, 8.2, ...]  # in feature frames
)

# Custom extractor for other formats
def my_extractor(json_file, fps, feat_stride):
    # Load your format, return list of durations in feature frames
    return [...]

ranges = compute_adaptive_ranges(
    json_file='data/my_format.json',
    num_levels=6,
    duration_extractor=my_extractor,
)
```

### CLI Usage

```bash
python -m libs.core.regression_ranges data/annotations.json --num-levels 6 --fps 30.0
```

## Loss Logger

```python
from libs.utils import LossLogger, compute_iou_stats

loss_logger = LossLogger(window_size=100)

for step, batch in enumerate(dataloader):
    losses = model.losses(...)
    iou_stats = compute_iou_stats(pred_offsets, gt_offsets)

    loss_logger.update(losses, extra_info={'num_pos': num_pos, **iou_stats})

    if step % 100 == 0:
        loss_logger.log_summary(step)

        # For tensorboard
        for name, val in loss_logger.get_tensorboard_dict().items():
            writer.add_scalar(name, val, step)
```

## Temporal Constraints

Post-process detections to enforce temporal rules.

```python
from libs.utils import TemporalConstraints, enforce_class_constraints

# Generic constraints
constraints = TemporalConstraints(
    min_gap={0: 5.0},  # Class 0 events at least 5s apart
    duration_range={0: (0.1, 2.0)},  # Duration limits for class 0
    max_overlap=0.3,  # Suppress overlapping detections
)
filtered = constraints.apply(results)

# Per-class convenience function
filtered = enforce_class_constraints(
    results,
    class_id=0,
    min_gap=5.0,
    duration_range=(0.1, 2.0),
)

# Merge close detections
from libs.utils import merge_close_detections
merged = merge_close_detections(results, merge_gap=0.5, class_id=0)
```

### Constraint Types

| Constraint | Description |
|------------|-------------|
| `min_gap` | Minimum time between same-class events |
| `duration_range` | (min, max) duration per class |
| `max_overlap` | Maximum IoU before suppression |
| `score_threshold` | Minimum score to keep |
