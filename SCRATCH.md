# Scratch Implementation Notes

This file tracks actionable decisions and follow-up work that came up while implementing Phase 1. The formal research plan remains in `design_doc.md`; this is the engineering scratchpad.

## Current Position

- Primary model is MobileViT-v1-S from timm: `mobilevit_s`.
- Current wrapper is `MobileViTSWithExits`.
- Architecture is MobileViT-v1-S plus five grafted exits and the original final classifier path.
- Exit placement:
  - Exit 1: after `stages.0`, 32 channels, LPH.
  - Exit 2: after `stages.1`, 64 channels, LPH.
  - Exit 3: after `stages.2`, 96 channels, GAH.
  - Exit 4: after `stages.3`, 128 channels, GAH.
  - Exit 5: after `stages.4`, 160 channels, GAH.
  - Final classifier: MobileViT `final_conv` plus `head`.
- Test harness verifies the final classifier output matches the underlying timm backbone output.
- Bounded Stage 0 -> Stage 1 -> Stage 2 -> eval smoke path has passed.

## Immediate Gates Before Attack Work

1. Run full Stage 0 on CIFAR-10 with pretrained MobileViT-S.
2. Run `eval_clean.py` on the Stage 0 checkpoint to confirm final classifier accuracy is moving toward a useful CIFAR-10 teacher.
3. Run Stage 1 joint supervision.
4. Run Stage 2 exit distillation.
5. Run `eval_clean.py` and compare per-exit accuracy to the Section 3.3 thresholds:
   - Exit 1: 70%.
   - Exit 2: 80%.
   - Exit 3: 87%.
   - Exit 4: 90%.
   - Final: around MobileViT-S CIFAR baseline.
6. Only start HybridDDAS once clean exit confidence and accuracy are meaningful.

## Cheap Checks To Run After Edits

```bash
.venv/bin/python -m pytest
.venv/bin/python -m py_compile models/*.py attacks/utils/*.py training/*.py eval/*.py
```

Bounded smoke path:

```bash
.venv/bin/python training/train_exits_stage0.py --epochs 1 --max-batches 2 --no-pretrained
.venv/bin/python training/train_exits_stage1.py --epochs 1 --max-batches 2
.venv/bin/python training/train_exits_stage2.py --epochs 1 --max-batches 2
.venv/bin/python eval/eval_clean.py --max-batches 2
```

`--no-pretrained` is only for wiring smoke tests. Real Stage 0 should use pretrained weights.

After full checkpoints finish:

```bash
.venv/bin/python eval/check_checkpoint.py models/mobilevit_s_cifar10_stage1_exp1.pt --max-batches 10 --output results/stage1_exp1_checkpoint_check.json
.venv/bin/python eval/check_checkpoint.py models/mobilevit_s_cifar10_stage2_exp1.pt --max-batches 10 --output results/stage2_exp1_checkpoint_check.json
.venv/bin/python eval/eval_clean.py --output results/clean_eval_stage2_exp1.json
```

## Training Strategy Questions

The current design uses:

1. Stage 0: adapt pretrained MobileViT-S to CIFAR-10 using final classifier loss.
2. Stage 1: train backbone plus exits with weighted cross-entropy.
3. Stage 2: freeze backbone and distill exits from the final classifier.

Add a cheaper baseline later:

- Freeze pretrained backbone.
- Train only final CIFAR classifier plus exit heads.
- Distill exits.
- Compare against the full Stage 0/1/2 adapted-backbone recipe.

This answers whether full adaptation is necessary or whether pretrained features are already enough for useful exits.

## Dataset Plan

Near term:

- CIFAR-10 first. It is cheap and exercises the full early-exit pipeline.
- Use our own non-exit/final-head MobileViT-S CIFAR-10 result as the fair baseline.

After CIFAR-10 works:

- Add an ImageNet-pretrained sanity check for the raw MobileViT-S final path.
- Consider Tiny-ImageNet as a second practical dataset before full ImageNet-scale work.

Benchmark notes:

- Published MobileViT-S ImageNet-1k reference is about 78.4% top-1 for MobileViT-v1-S.
- Do not assume CIFAR-10 or Tiny-ImageNet early-exit thresholds from published results unless the setup matches. Use our final-head baseline and report exit degradation relative to it.
- Tiny-ImageNet thresholds should be defined after measuring the final-head baseline, not copied from CIFAR-10.

## Later Model Targets

MobileViT-v2 should be added later as a secondary hybrid target, not as a replacement for v1.

Reason:

- MobileViT-v1 uses the original MobileViT block design.
- MobileViT-v2 replaces multi-head self-attention with separable self-attention.
- That difference is directly relevant to whether transformer-stage attention behavior changes DDAS-style entropy manipulation.

Likely additions:

```text
models/mobilevit_v2_exits.py
configs/mobilevit_v2.yaml
```

Before implementing v2:

- Verify timm model name and pretrained weights.
- Inspect stage outputs and channel dimensions.
- Match exit locations to the v1 wrapper as closely as possible.
- Reuse the shared training/eval harness.

## Attack Phase Readiness

Do not implement attack code until:

- CIFAR-10 final classifier is trained enough to be a useful teacher.
- Exit heads show non-random clean accuracy.
- Sequential threshold calibration produces meaningful thresholds above the floor.
- `eval_clean.py` reports stable six-output accuracy.

Attack-specific cheap tests to add when that phase starts:

- HybridDDAS increases exit entropy versus clean inputs.
- HybridDDAS preserves final prediction for most samples.
- Perturbation stays inside the L-inf budget.
- APR computation handles sequential exit decisions correctly.
