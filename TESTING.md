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
5. Artifact check: checkpoint strict-load plus bounded per-output accuracy via `eval/check_checkpoint.py`.

Full Stage 0/1/2 training should only run after these gates pass.

## Current Command

```bash
.venv/bin/python -m pytest
```

## Bounded Training Checks

The stage scripts support `--max-batches` so agents can exercise the real training path without launching a full run:

```bash
.venv/bin/python training/train_exits_stage0.py --epochs 1 --max-batches 2 --no-pretrained
.venv/bin/python training/train_exits_stage1.py --epochs 1 --max-batches 2
.venv/bin/python training/train_exits_stage2.py --epochs 1 --max-batches 2
.venv/bin/python eval/eval_clean.py --max-batches 2
```

Stage 1 requires the Stage 0 checkpoint, and Stage 2 requires the Stage 1 checkpoint. The `--no-pretrained` flag is only for smoke tests that validate wiring without downloading ImageNet weights; real Stage 0 should use the default pretrained backbone. Full Stage 0/1/2 training should only run after the pytest harness passes.

## Artifact Checks

Training scripts save a checkpoint and a JSON sidecar next to it, for example:

```text
models/mobilevit_s_cifar10_stage1.pt
models/mobilevit_s_cifar10_stage1.pt.json
```

After a checkpoint finishes, run a bounded sanity check:

```bash
.venv/bin/python eval/check_checkpoint.py models/mobilevit_s_cifar10_stage1_exp1.pt --max-batches 10 --output results/stage1_exp1_checkpoint_check.json
```

Clean eval writes structured results by default:

```bash
.venv/bin/python eval/eval_clean.py --output results/clean_eval_stage2_exp1.json
```
