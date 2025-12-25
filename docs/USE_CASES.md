# Use Cases Guide

This guide helps you choose the right configuration for your specific use case.

## Decision Tree

```
Start
  │
  ├─ What type of events are you detecting?
  │   ├─ Point/instant events (duration ≈ 0) → SnapFormer
  │   ├─ Segments with noisy labels → TBTFormer
  │   └─ Segments with clean labels → ActionFormer (LocPointTransformer)
  │
  ├─ Do you have multiple GPUs?
  │   ├─ Yes → Use train_ddp.py with DDP
  │   └─ No  → Use train.py (single GPU)
  │
  ├─ Do you have PyTorch 2.0+?
  │   ├─ Yes → Enable Flash Attention
  │   └─ No  → Use standard attention
  │
  ├─ Are your sequences longer than 2048?
  │   ├─ Yes → Use RoPE + Flash Attention (critical)
  │   └─ No  → RoPE still recommended
  │
  ├─ Is training speed critical?
  │   ├─ Yes → Use RMSNorm + AMP
  │   └─ No  → LayerNorm is fine
  │
  └─ Is model quality critical?
      ├─ Yes → Use SwiGLU
      └─ No  → Standard MLP is fine
```

## Architecture Selection

| Architecture | Use Case | Output |
|--------------|----------|--------|
| `LocPointTransformer` | Standard action segmentation | Segments (start, end) |
| `SnapFormer` | Point/instant events (snaps, clicks, impacts) | Points (time only) |
| `TBTFormer` | Noisy annotations, uncertain boundaries | Segments with uncertainty |

## Common Scenarios

### 1. Quick Prototyping

**Goal**: Fast iteration, validate ideas quickly

```yaml
model:
  backbone_type: convTransformer  # Use v1 for simplicity
```

```bash
# Single GPU, no frills
python train.py configs/thumos_i3d.yaml --output prototype
```

### 2. Production Training

**Goal**: Best quality, reasonable training time

```yaml
model:
  backbone_type: convTransformerv2
  backbone:
    use_rope: true
    use_flash_attn: true
    use_swiglu: true
    use_rms_norm: true
```

```bash
# Multi-GPU with AMP
torchrun --nproc_per_node=4 train_ddp.py config.yaml --amp --output prod
```

### 3. Limited GPU Memory

**Goal**: Train with limited VRAM (8-12GB)

```yaml
model:
  backbone_type: convTransformerv2
  backbone:
    use_flash_attn: true  # Reduces memory
    use_rope: true
    use_swiglu: false     # Saves memory
    use_rms_norm: true

loader:
  batch_size: 1           # Reduce batch size
```

```bash
# Use gradient accumulation to compensate
torchrun --nproc_per_node=2 train_ddp.py config.yaml \
    --amp --accum-steps 4 --output limited_mem
```

### 4. Long Video Sequences

**Goal**: Handle videos with T > 4096 frames/features

```yaml
model:
  backbone_type: convTransformerv2
  backbone:
    max_len: 8192         # Increase max length
    use_rope: true        # Critical - extrapolates to any length
    use_flash_attn: true  # Critical - O(T) memory
    use_rms_norm: true
    use_abs_pe: false     # Don't use with RoPE
```

### 5. Inference/Deployment

**Goal**: Fast inference, low latency

```yaml
model:
  backbone_type: convTransformerv2
  backbone:
    n_kv_head: 2          # GQA for faster inference
    use_rope: true
    use_flash_attn: true
    use_swiglu: false     # Faster
    use_rms_norm: true
```

### 6. Research/Ablation Studies

**Goal**: Compare v1 vs v2 components

```yaml
# Config A: v1 baseline
model:
  backbone_type: convTransformer
  backbone:
    use_abs_pe: true

# Config B: v2 with only Flash Attention
model:
  backbone_type: convTransformerv2
  backbone:
    use_flash_attn: true
    use_rope: false
    use_swiglu: false
    use_rms_norm: false
    use_abs_pe: true

# Config C: v2 with only RoPE
model:
  backbone_type: convTransformerv2
  backbone:
    use_flash_attn: false
    use_rope: true
    use_swiglu: false
    use_rms_norm: false
    use_abs_pe: false

# Config D: Full v2
model:
  backbone_type: convTransformerv2
  backbone:
    use_flash_attn: true
    use_rope: true
    use_swiglu: true
    use_rms_norm: true
```

### 7. Large-Scale Training (Cluster)

**Goal**: Train on multiple nodes

```bash
# Node 0 (master)
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=0 \
    --master_addr=$MASTER_ADDR --master_port=29500 \
    train_ddp.py config.yaml --amp --accum-steps 2

# Node 1-3
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=$RANK \
    --master_addr=$MASTER_ADDR --master_port=29500 \
    train_ddp.py config.yaml --amp --accum-steps 2
```

### 8. Point/Instant Event Detection (SnapFormer)

**Goal**: Detect events with near-zero duration (snaps, clicks, keystrokes, impacts)

```yaml
model:
  meta_arch: "SnapFormer"
  fpn_type: "cs_fpn"              # Cross-scale FPN recommended
  backbone_type: convTransformerv2
  backbone:
    use_rope: true
    use_flash_attn: true
```

**Examples**: Football snap detection, keystroke detection, ball impact detection

**Key differences from ActionFormer**:
- Outputs points (single timestamp) instead of segments
- Uses Gaussian heatmap regression instead of boundary regression
- Inference via peak detection

### 9. Noisy Annotations / Uncertain Boundaries (TBTFormer)

**Goal**: Handle datasets with annotation noise or ambiguous action boundaries

```yaml
model:
  meta_arch: "TBTFormer"
  fpn_type: "cs_fpn"
  reg_max: 16                     # Distribution bins
  dfl_weight: 0.25                # DFL loss weight
  backbone_type: convTransformerv2
  backbone:
    n_head: 16                    # Scaled backbone (optional)
    use_rope: true
    use_flash_attn: true
```

**Examples**: Crowdsourced annotations, ambiguous action transitions, fine-grained actions

**Key differences from ActionFormer**:
- Predicts probability distribution over boundary locations
- Uses Distribution Focal Loss for smoother learning
- Better handles label noise (±1-2 frame errors)

## Dataset-Specific Recommendations

### THUMOS14

- Short videos, dense annotations
- Moderate sequence lengths

```yaml
model:
  backbone_type: convTransformerv2
  backbone:
    n_embd: 256
    n_head: 4
    arch: [2, 2, 5]
    mha_win_size: [-1, -1, -1, -1, -1, -1]  # Full attention OK
    use_rope: true
    use_flash_attn: true
    use_swiglu: true
    use_rms_norm: true
```

### ActivityNet

- Longer videos, sparser annotations
- Benefits from local attention

```yaml
model:
  backbone_type: convTransformerv2
  backbone:
    n_embd: 256
    n_head: 4
    arch: [2, 2, 5]
    mha_win_size: [9, 9, 9, 9, 9, -1]  # Local + global
    use_rope: true
    use_flash_attn: true
    use_swiglu: true
    use_rms_norm: true
```

### EPIC-Kitchens

- Egocentric videos, fine-grained actions
- Higher resolution features

```yaml
model:
  backbone_type: convTransformerv2
  backbone:
    n_embd: 384           # Larger embedding
    n_head: 6
    arch: [2, 2, 6]       # More layers
    use_rope: true
    use_flash_attn: true
    use_swiglu: true
    use_rms_norm: true
```

### Ego4D

- Very long videos, multiple feature types
- Critical to use memory-efficient attention

```yaml
model:
  backbone_type: convTransformerv2
  backbone:
    n_embd: 768
    n_head: 12
    arch: [2, 2, 7]
    max_len: 4096
    use_rope: true        # Critical for length
    use_flash_attn: true  # Critical for memory
    use_swiglu: true
    use_rms_norm: true
```

## Hardware-Specific Recommendations

### NVIDIA A100 / H100

```bash
# Maximum utilization
torchrun --nproc_per_node=8 train_ddp.py config.yaml \
    --amp --accum-steps 1
```

- Full v2 features
- Large batch sizes
- AMP always

### NVIDIA V100

```bash
# Good balance
torchrun --nproc_per_node=4 train_ddp.py config.yaml \
    --amp --accum-steps 2
```

- Full v2 features
- Moderate batch sizes
- AMP recommended

### NVIDIA RTX 3090 / 4090

```bash
# Consumer GPU optimization
torchrun --nproc_per_node=2 train_ddp.py config.yaml \
    --amp --accum-steps 4
```

- Flash Attention critical
- Smaller batch sizes
- AMP required

### CPU-Only (Not Recommended)

```bash
# Only for debugging
python train.py config.yaml --output cpu_debug
```

- Disable Flash Attention
- Very slow, not for actual training

## Performance Comparison

| Configuration | Training Time | Memory | Quality |
|---------------|---------------|--------|---------|
| v1 baseline | 1.0x | 1.0x | Baseline |
| v2 (Flash only) | 0.85x | 0.75x | Same |
| v2 (RMSNorm only) | 0.93x | 1.0x | Same |
| v2 (SwiGLU only) | 1.05x | 1.05x | Better |
| v2 (all features) | 0.80x | 0.75x | Better |
| v2 + AMP | 0.45x | 0.50x | Same |
| v2 + AMP + 4 GPU | 0.12x | 0.50x | Same |

*Relative to v1 single GPU baseline*

## Troubleshooting Guide

| Problem | Solution |
|---------|----------|
| OOM on long sequences | Enable Flash Attention, reduce batch size |
| Slow training | Enable AMP, use DDP, check data loading |
| Poor convergence | Check learning rate scaling, try without SwiGLU |
| NaN losses | Reduce learning rate, check gradient clipping |
| Checkpoint incompatible | v1 and v2 checkpoints are not interchangeable |
