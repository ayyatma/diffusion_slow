# Testing Harness

Use the project venv for all checks:

```bash
source .venv/bin/activate
python -m pytest
```

The harness is intentionally cheap. It should run before full CIFAR training and after every model, loss, or evaluation edit.

## Gates

1. Static/runtime unit tests: config parsing, entropy math, threshold decisions, exit-head shapes.
2. Model contract tests: MobileViT-S returns five exit logits plus one final classifier output.
3. Gradient-flow test: joint exit loss backpropagates through every trainable parameter.
4. Stage-loss test: distillation KL loss is finite and backpropagates.

Full Stage 0/1/2 training should only run after these gates pass.

## Current Command

```bash
.venv/bin/python -m pytest
```

## Bounded Training Checks

The stage scripts support `--max-batches` so agents can exercise the real training path without launching a full run:

```bash
.venv/bin/python training/train_exits_stage0.py --max-batches 2
.venv/bin/python training/train_exits_stage1.py --max-batches 2
.venv/bin/python training/train_exits_stage2.py --max-batches 2
.venv/bin/python eval/eval_clean.py --max-batches 2
```

Stage 1 requires the Stage 0 checkpoint, and Stage 2 requires the Stage 1 checkpoint. Full Stage 0/1/2 training should only run after the pytest harness passes.
