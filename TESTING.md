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
