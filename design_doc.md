# Design Document: Resource-Exhaustion Adversarial Attacks on Hybrid CNN-Transformer Early-Exit Networks

**Working Title:** *Attacking the Efficiency of Hybrid Early-Exit Networks: Resource-Exhaustion Adversarial Threats on Edge-Deployed CNN-Transformer Models*

**Author:** Mohammed Ayyat  
**Advisor:** Prof. Tamer Nadeem  
**Lab:** MuSIC Lab, Virginia Commonwealth University  
**Status:** Pre-implementation design v1.1 — implementation constants added  
**Target Venue:** IEEE Transactions on Mobile Computing (TMC) or IEEE PerCom 2026  
**Timeline:** 8 weeks to dissertation-ready results; full paper submission post-graduation

---

## 1. Motivation and Research Gap

### 1.1 The Problem

Your DDAS line of work established a new threat model: adversaries can degrade the *efficiency* of early-exit networks (EENs) without degrading prediction accuracy, by manipulating input entropy so that samples bypass early exits and force computation to deeper layers. This "denial-of-efficiency" attack increases latency and power consumption while remaining invisible to accuracy-based monitoring.

DDAS was validated on standard CNN backbones (ResNet variants) with MobileNetV2. This is a legitimate reviewer objection: these are not the models operators deploy in 2025. The current edge-realistic architectures are **hybrid CNN-transformer models** — specifically MobileViT, MobileViTv2, and MobileNetV4 — which mix convolutional stages with self-attention blocks.

### 1.2 Why This Is a New Problem, Not an Extension

Hybrid architectures introduce a structurally distinct early-exit surface:

- **CNN-stage exits** (after early inverted residual blocks): feature maps are spatially rich but semantically shallow. Confidence signals here are similar to pure CNN behavior — entropy manipulation should work analogously to DDAS.
- **Transformer-stage exits** (after MobileViT attention blocks): feature representations encode global context. Prior work on accuracy attacks shows ViTs are *more robust* than CNNs because their features contain less low-level exploitable structure. **Nobody has studied whether this extends to resource-exhaustion attacks.** It may not — entropy in a transformer-stage classifier depends on attention weight distribution, which is a different signal than softmax confidence in a CNN exit.

This architectural duality is the core scientific question: **does the hybrid nature of MobileViT create differential vulnerability across exit stages, and can an attacker exploit that asymmetry?**

### 1.3 Novelty Claims (Defensible at Submission)

1. First study of resource-exhaustion adversarial attacks on hybrid CNN-transformer early-exit networks.
2. First characterization of differential exit-stage vulnerability in hybrid EENs (CNN exits vs. transformer exits under the same attack).
3. First denial-of-efficiency evaluation on a hardware-realistic edge target (MobileViT-S on Jetson TX2) using a modern architecture accepted as current SOTA for edge deployment.
4. Empirical finding: whether transformer-stage exits dampen or amplify resource-exhaustion attacks relative to CNN-stage exits (either result is novel and publishable).

---

## 2. Target Architecture

### 2.1 Primary Target: MobileViT-S

**Why MobileViT-S specifically:**
- 5.6M parameters, ~2 GFLOPs — runs on Jetson TX2 without quantization
- Publicly available pretrained weights (timm, HuggingFace)
- Three natural stage boundaries, each a clean exit placement point
- Hybrid structure directly motivates the differential vulnerability analysis
- ECCV/ICLR-published, widely cited — no reviewer will question relevance

**Architecture summary (MobileViT-S):**

```
Input (224×224)
│
├── Stem: 3×3 Conv, stride 2 → 16 channels
│
├── Stage 1: MV2 blocks × 1, stride 2 → 32ch  [EXIT 1 — pure CNN]
│
├── Stage 2: MV2 blocks × 3, stride 2 → 64ch  [EXIT 2 — pure CNN]
│
├── Stage 3: MV2 + MobileViT block (dim=96)    [EXIT 3 — CNN+Transformer hybrid]
│   stride 2 → 96ch, 28×28 feature maps
│
├── Stage 4: MV2 + MobileViT block (dim=120)   [EXIT 4 — Transformer-dominant]
│   stride 2 → 128ch, 14×14 feature maps
│
├── Stage 5: MV2 + MobileViT block (dim=144)   [EXIT 5 — Transformer-dominant]
│   → 160ch, 7×7 feature maps
│
└── Head: Conv 1×1 → 640ch → GlobalAvgPool → FC(1000)
```

**Natural exit placement:** After each stage boundary — 5 possible exits, 3 of which are post-transformer. This gives you a rich analysis of CNN-exit vs. transformer-exit differential behavior.

### 2.2 Secondary Target: MobileNetV4-Conv-S

- Pure CNN, ECCV 2024, Pareto-optimal across all edge hardware
- Serves as a control: if MobileNetV4 (no transformer) shows similar vulnerability to MobileViT's CNN-stage exits, you validate that the transformer stages are the variable.
- 3.8M parameters, runs cleanly on Jetson TX2

### 2.3 Legacy Baselines (Keep for Continuity with DDAS)

- ResNet-56 (existing DDAS baseline — continuity with TMC paper)
- MobileNetV2 (existing RACENet/DDAS baseline)

Keeping these lets you show the attack generalizes across generations of architecture, not just that it works on the new model.

---

## 3. Early-Exit Grafting Strategy

### 3.1 Exit Head Design (Based on LGViT Recipe)

LGViT (ACM MM 2023) identified the core problem: a single exit head design fails across hybrid architectures because early CNN-stage exits and late transformer-stage exits have fundamentally different feature characteristics.

**Solution — Heterogeneous exit heads:**

| Exit Stage | Feature Type | Exit Head Architecture |
|---|---|---|
| Exit 1–2 (CNN) | Spatial, local | Local Perception Head (LPH): lightweight 3×3 depthwise conv → BN → ReLU → GlobalAvgPool → FC |
| Exit 3–5 (Transformer) | Global, patch-based | Global Aggregation Head (GAH): LayerNorm → Linear projection → CLS token aggregation → FC |

This is not novel — LGViT did it for DeiT and Swin. Your contribution is applying it to MobileViT's specific hybrid structure and then *attacking* it, which LGViT never does.

### 3.2 Training Recipe for Grafted Exits

**Two-stage training (mirrors LGViT):**

**Stage 1 — Backbone fine-tuning with exit supervision:**
- Loss: weighted sum of cross-entropy at each exit + final layer
- Weight schedule: higher weight on final exit (backbone preservation), lower on early exits
- Epochs: 30–50 on CIFAR-100 or ImageNet-100 subset
- Optimizer: AdamW, lr=1e-4, cosine decay

**Stage 2 — Exit distillation:**
- Freeze backbone weights
- Train exit classifiers using knowledge distillation from final layer
- Loss: KL divergence (soft labels from final exit) + cross-entropy (hard labels)
- Epochs: 20–30

**Week 1 feasibility test:** Run Stage 1 on CIFAR-10 for 10 epochs. If exit accuracy at Exit 3+ is within 5% of final layer accuracy, the grafting is working. If not, adjust distillation temperature before proceeding.

### 3.3 Accuracy Targets for Grafted Exits (Acceptance Threshold)

| Exit | Min Acceptable Top-1 (CIFAR-10) |
|---|---|
| Exit 1 | 70% |
| Exit 2 | 80% |
| Exit 3 | 87% |
| Exit 4 | 90% |
| Exit 5 (final) | ~93% (MobileViT-S baseline) |

If Exit 1–2 accuracy is below threshold, the exits are too shallow to be a meaningful attack surface — adjust head architecture before proceeding to attack phase.

---

## 4. Attack Design

### 4.1 Threat Model

**Attacker goal:** Force inputs to bypass early exits and propagate to deeper layers, increasing inference latency and energy consumption without changing the final prediction.

**Attacker knowledge:**
- White-box: full access to model weights, exit thresholds, gradient information
- Black-box: access to inference timing only (surrogate distillation via timing side-channel, as in DDAS TMC)

**Constraints:**
- Perturbation budget: L∞ ε = 8/255 (standard for image classifiers)
- Perturbation must be imperceptible (PSNR > 40 dB target)
- Final prediction must not change (attack is not an accuracy attack)

### 4.2 Attack Mechanism: Entropy Manipulation

The mechanism is inherited from DDAS: craft perturbations that increase prediction entropy at exit classifiers so the confidence threshold is never met, forcing the sample to propagate forward.

**For CNN-stage exits (Exit 1–2):**
- Gradient-based entropy maximization on softmax output of LPH
- Identical in principle to DDAS — use existing implementation as baseline

**For Transformer-stage exits (Exit 3–5):**
- Entropy target is on the GAH output, but gradient flows through the attention mechanism
- Attention maps in MobileViT operate on unfolded patches — perturbation must diffuse across patch boundaries to disrupt global aggregation
- This is architecturally harder than CNN-exit manipulation: **the patch-based global processing may dilute local perturbations**, which is the core scientific question

### 4.3 Attack Algorithm

```
Algorithm: HybridDDAS
Input: image x, model f with exits {e1,...,e5}, 
       perturbation budget ε, iterations T
Output: adversarial example x_adv

1. Initialize δ ~ Uniform(-ε, ε)
2. For t = 1 to T:
   a. Forward pass x + δ through f
   b. For each active exit ei:
      - Compute entropy H(ei(x+δ))
      - Compute confidence gap: Δi = threshold_i - max_prob(ei)
   c. Loss = -Σ_i w_i · H(ei(x+δ))           [maximize entropy at each exit]
            + λ · ||δ||_2                       [perturbation size penalty]
            + μ · max(0, pred_change_penalty)   [preserve final prediction]
   d. δ ← δ + α · sign(∇_δ Loss)
   e. δ ← clip(δ, -ε, ε)
3. Return x + δ
```

**Key design choice — stage-aware loss weighting (w_i):**
- Experiment A (uniform): equal weight on all exits
- Experiment B (CNN-first): higher weight on exits 1–2, lower on 3–5
- Experiment C (transformer-first): higher weight on exits 3–5

This weighting experiment directly tests whether attacking transformer exits is more or less efficient than attacking CNN exits — the novel finding.

### 4.4 Black-Box Extension

Use the surrogate distillation approach from DDAS TMC:
1. Query target model with random inputs, collect timing responses
2. Train surrogate model (same architecture, distilled from timing labels)
3. Run white-box attack on surrogate, transfer to target

For MobileViT, timing side-channel is particularly rich because CNN-stage exits (fast) vs. transformer-stage exits (slower due to attention computation) have measurably different latency profiles. This makes exit-level inference from timing more accurate than in pure CNN models.

---

## 5. Evaluation Plan

### 5.1 Datasets

| Dataset | Why |
|---|---|
| CIFAR-10 | Continuity with DDAS baseline; fast iteration |
| CIFAR-100 | Harder classification → more meaningful exit distributions |
| ImageNet-100 (subset) | Reviewer-credible; tests at realistic resolution |
| WISDM (activity recognition) | Time-series; continuity with RACENet/CAFED; shows non-vision generality |

### 5.2 Metrics

**Primary (attack effectiveness):**
- **Attack Persistence Rate (APR):** percentage of inputs forced past all early exits (matches DDAS TMC metric for direct comparison)
- **Latency increase (%):** mean inference time under attack vs. clean input, measured on Jetson TX2
- **Power increase (%):** mean power draw (mW) under attack vs. clean, measured on Jetson TX2
- **Accuracy preservation (%):** final-layer accuracy under attack (should remain within 1–2% of clean)

**Secondary (perturbation quality):**
- **PSNR:** should exceed 40 dB (DDAS TMC achieves 55.4 — set this as target)
- **SSIM:** supplementary metric

**Novel metrics (exit-stage analysis):**
- **Stage-level APR:** APR broken down by CNN-stage exits vs. transformer-stage exits separately
- **Attention disruption score:** mean entropy increase at transformer-stage exits per unit perturbation magnitude (efficiency of attack per stage type)

### 5.3 Baselines

| Baseline | Purpose |
|---|---|
| DDAS (CNN targets) | Direct predecessor — shows architectural generalization |
| ILFO | Prior work, already beaten in DDAS TMC |
| DeepSloth | Prior work, already beaten in DDAS TMC |
| No-attack (clean) | Lower bound |
| Random noise (ε=8/255) | Sanity check |

### 5.4 Ablation Studies

1. **Stage weighting ablation:** Experiments A/B/C from Section 4.3 — CNN-first vs. transformer-first vs. uniform weighting
2. **Exit head architecture ablation:** LPH-only vs. GAH-only vs. heterogeneous — confirms the heterogeneous design is necessary
3. **Perturbation budget sweep:** ε ∈ {4/255, 8/255, 16/255} — shows attack-stealth tradeoff
4. **Attack iterations sweep:** T ∈ {10, 20, 50} — convergence analysis

### 5.5 Hardware Evaluation (Jetson TX2)

All latency and power measurements on Jetson TX2 in maximum-performance mode (matching DDAS TMC methodology exactly):

- Baseline inference: 100 clean samples × 5 runs, report mean ± std
- Under attack: same 100 adversarial samples × 5 runs
- Power measured via `tegrastats` at 100ms polling interval
- Batch size = 1 (realistic edge deployment)

---

## 6. Defense Evaluation

Keep this section lean — the defense story from DDAS TMC already establishes adversarial retraining as a partial mitigation. For this paper, evaluate two defenses:

1. **Adversarial retraining (from DDAS):** apply iteratively to hybrid exits; report how many iterations before convergence on MobileViT
2. **Exit threshold hardening:** raise confidence threshold at transformer-stage exits (simple, deployable) — measure APR reduction vs. clean accuracy cost

The defense section should be 1–2 tables, not a full contribution. The main contribution is the attack analysis.

---

## 7. Dissertation Integration Plan

### What Goes in the Dissertation (8-week target)

**Chapter outline addition:**
- Chapter X: *Resource-Exhaustion Attacks on Hybrid Early-Exit Architectures*
  - Section 1: Motivation (CNN-transformer hybrid attack surface)
  - Section 2: MobileViT-S exit grafting (methodology + Stage 1 accuracy results)
  - Section 3: HybridDDAS attack algorithm
  - Section 4: Preliminary results — CIFAR-10, latency/power on Jetson TX2, APR vs. DDAS baseline
  - Section 5: Stage-level vulnerability analysis (key novel finding)
  - Section 6: Proposed extension (ImageNet-100, full ablations, black-box — for journal version)

**Minimum viable results for dissertation inclusion:**
- Grafted MobileViT-S with 5 exits, accuracy within threshold on CIFAR-10
- APR, latency increase, power increase on CIFAR-10 (white-box only)
- Stage-level APR breakdown showing CNN vs. transformer exit differential
- One ablation (stage weighting)

**What defers to journal version:**
- ImageNet-100 evaluation
- Black-box attack + surrogate distillation
- MobileNetV4-Conv-S secondary target
- Full defense section
- WISDM time-series evaluation

---

## 8. Week-by-Week Timeline

| Week | Tasks | Go/No-Go Gate |
|---|---|---|
| **1** | Set up MobileViT-S in timm. Graft 5 exits using LGViT recipe (LPH for exits 1–2, GAH for exits 3–5). Run Stage 1 training on CIFAR-10 for 10 epochs. | Exit 3+ accuracy ≥ 87% on CIFAR-10. If not, adjust distillation temperature and add 3–4 days. |
| **2** | Complete Stage 2 distillation. Validate all 5 exits meet accuracy thresholds. Set clean inference baseline on Jetson TX2 (latency, power per exit). | All exits meet thresholds. Jetson TX2 baseline numbers recorded. |
| **3** | Implement HybridDDAS (extend DDAS codebase). Run Experiment A (uniform weighting) on CIFAR-10 white-box. Collect APR, latency, power. | APR > 70% on CNN-stage exits (lower bar; transformer exits may be harder). |
| **4** | Run Experiments B and C (stage weighting ablations). Compute stage-level APR breakdown. This is the core scientific result — analyze and interpret carefully. | Clear differential between CNN-stage and transformer-stage APR. If not, investigate perturbation diffusion across patch boundaries. |
| **5** | Extend to CIFAR-100. Run perturbation budget sweep (ε ablation). Add MobileNetV4-Conv-S as control target. | CIFAR-100 results consistent with CIFAR-10 trend. |
| **6** | Hardware evaluation on Jetson TX2 (full latency/power table). Add DDAS CNN-baseline comparison row. Run adversarial retraining defense (2–3 iterations). | Latency increase measurable (target ≥ 30%). Power increase measurable (target ≥ 50%). |
| **7** | Write dissertation chapter (Sections 1–5). Generate all figures: exit accuracy bar chart, APR by stage heatmap, latency/power comparison table, perturbation budget tradeoff curve. | Draft chapter complete. |
| **8** | Revise chapter. Identify gaps for journal version. Write "future work" section covering black-box extension and ImageNet-100 evaluation. | Chapter ready for advisor review. |

---

## 9. Codebase Plan

### 9.1 Repository Structure

```
hybrid-ddas/
├── models/
│   ├── mobilevit_s_exits.py      # MobileViT-S with grafted exits
│   ├── mobilenetv4_exits.py      # MobileNetV4 with grafted exits  
│   └── exit_heads.py             # LPH and GAH implementations
├── training/
│   ├── train_exits_stage1.py     # Backbone fine-tuning with exit supervision
│   └── train_exits_stage2.py     # Exit distillation
├── attacks/
│   ├── hybrid_ddas.py            # Main attack implementation
│   ├── baselines/
│   │   ├── ilfo.py               # From DDAS codebase
│   │   └── deepsloth.py          # From DDAS codebase
│   └── utils/
│       ├── entropy.py            # Entropy computation utilities
│       └── stage_analysis.py     # Stage-level APR computation
├── eval/
│   ├── eval_clean.py             # Baseline accuracy and latency
│   ├── eval_attack.py            # APR, latency, power under attack
│   └── eval_defense.py           # Adversarial retraining loop
├── hardware/
│   └── jetson_profiler.py        # tegrastats wrapper for power/latency
├── configs/
│   ├── mobilevit_s.yaml          # Model + exit head config
│   └── attack_config.yaml        # Attack hyperparameters
└── notebooks/
    └── analysis.ipynb            # Figure generation
```

### 9.2 Key Dependencies

```
timm >= 0.9.0               # MobileViT-S pretrained weights
torch >= 2.0                # Gradient computation for attacks
torchvision                 # Datasets
numpy, pandas               # Results processing
matplotlib, seaborn         # Figures
```

### 9.3 Starting Point from DDAS Codebase

Reuse directly:
- `entropy.py` — entropy computation and threshold logic
- `ilfo.py`, `deepsloth.py` — baseline attack implementations
- `jetson_profiler.py` — hardware measurement wrapper
- Attack loop structure from `ddas_attack.py` — extend, don't rewrite

New code required:
- `mobilevit_s_exits.py` — exit head grafting (primary new implementation)
- `hybrid_ddas.py` — stage-aware loss weighting (extend existing attack loop)
- `stage_analysis.py` — new metric: per-stage APR breakdown

---

## 10. Paper Outline (Journal Version)

**Target:** IEEE Transactions on Mobile Computing (TMC) — same venue as DDAS follow-on  
**Target length:** 13–15 pages double-column

| Section | Content |
|---|---|
| Abstract | Threat model, hybrid EEN target, key results (APR, latency, power), differential finding |
| I. Introduction | DDAS recap (2 para), why hybrid architectures change the attack surface, contributions list |
| II. Background | Early-exit networks; hybrid CNN-transformer models (MobileViT); existing resource-exhaustion attacks |
| III. Threat Model | Attacker capabilities, goals, constraints; white-box and black-box scenarios |
| IV. Exit Grafting for Hybrid Models | LPH/GAH design, training recipe, accuracy validation |
| V. HybridDDAS Attack | Algorithm, stage-aware loss, convergence analysis |
| VI. Experimental Setup | Models, datasets, hardware, metrics, baselines |
| VII. Results | Main APR/latency/power table; stage-level breakdown (key novel result); ablations |
| VIII. Black-Box Extension | Surrogate distillation via timing side-channel; transfer results |
| IX. Defenses | Adversarial retraining on hybrid exits; threshold hardening |
| X. Discussion | Implications for edge security; why transformer stages may not protect against resource attacks |
| XI. Conclusion | Summary, limitations, future work |

---

## 11. Contribution Statement (for paper/dissertation)

> This paper presents the first systematic study of resource-exhaustion adversarial attacks on hybrid CNN-transformer early-exit networks deployed on edge hardware. We graft early exits onto MobileViT-S using heterogeneous exit heads and demonstrate that the HybridDDAS attack — an entropy-manipulation perturbation method extended from DDAS — forces samples past all early exits with an Attack Persistence Rate of [X]%, increasing inference latency by [Y]% and power consumption by [Z]% on the Jetson TX2 edge platform. Critically, we find that transformer-stage exits exhibit [higher/lower/comparable] vulnerability to resource-exhaustion attacks compared to CNN-stage exits, revealing that architectural robustness properties established for accuracy attacks do not straightforwardly transfer to the denial-of-efficiency threat model. Our results extend the DDAS threat model to the current generation of edge-realistic architectures and motivate exit-stage-aware defense mechanisms.

*Fill in X/Y/Z after experiments. Either direction of the transformer finding is publishable — the analysis is the contribution.*

---

## 12. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Exit grafting accuracy too low on MobileViT transformer stages | Medium | Increase distillation epochs; try temperature scaling; fall back to accuracy-fine-tuned exits without distillation |
| Attack shows no differential between CNN and transformer stages | Low-Medium | Still publishable as "hybrid architecture robustness equivalence under resource attacks" — reframe, not abandon |
| Jetson TX2 OOM for MobileViT-S at batch=1 | Low | Use MobileViT-XXS (2.3M params) as fallback; results still valid |
| Timeline slips past 8 weeks | Medium | Dissertation chapter can omit CIFAR-100 and MobileNetV4 — CIFAR-10 + Jetson TX2 is sufficient for initial inclusion |
| LGViT exit recipe doesn't transfer to MobileViT's shallower stages | Medium | Simplify LPH to GlobalAvgPool + FC only (no conv) for exits 1–2; transformer-stage GAH is the priority |

---

## 13. Implementation Constants and Underspecified Decisions

This section exists specifically for agentic implementation. Every value here is a concrete default — no guessing required. Where a sweep is planned, the default is the middle value. All values are changeable via `configs/attack_config.yaml` and `configs/mobilevit_s.yaml` without touching source code.

---

### 13.1 Input Resolution and Dataset Handling

**MobileViT-S expects 256×256 input.** CIFAR-10/100 are 32×32 — they must be resized.

```python
# Standard transforms for all datasets
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(256, padding=16),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # ImageNet stats
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

**Dataset splits:**

| Dataset | Train | Val (threshold calibration) | Test (attack eval) |
|---|---|---|---|
| CIFAR-10 | 45,000 | 5,000 (from train split) | 10,000 (standard test) |
| CIFAR-100 | 45,000 | 5,000 | 10,000 |
| ImageNet-100 | 130,000 | 5,000 | 5,000 |

**ImageNet-100 subset:** Use the first 100 classes alphabetically from ImageNet-1k. This is a standard reproducible subset — do not randomly sample classes, as results must be reproducible without a fixed seed file.

**WISDM:** Use the standard 70/30 train/test split. Input is a 1D time-series window of 200 timesteps × 3 axes, reshaped to 3×1×200 and processed through a 1D-adapted MobileViT. Defer this to journal version — do not implement in initial agentic run.

---

### 13.2 MobileViT-S Backbone Loading

```python
import timm

# Load pretrained MobileViT-S
# timm model name: 'mobilevit_s'
backbone = timm.create_model('mobilevit_s', pretrained=True, num_classes=0)
# num_classes=0 removes the head — we attach our own exit heads + final head
```

**Pretrained weights source:** timm default (ImageNet-1k). For CIFAR experiments, fine-tune the backbone for 10 epochs before grafting exits (Stage 0 in training recipe — add this before Stage 1).

**Stage 0 — Backbone adaptation to CIFAR (new, not in Section 3.2):**
```
Optimizer: AdamW, lr=1e-4, weight_decay=1e-4
Scheduler: CosineAnnealingLR, T_max=10
Epochs: 10
Batch size: 128
Loss: CrossEntropyLoss on final FC head only
Purpose: Adapt ImageNet-pretrained weights to CIFAR resolution/distribution
         before grafting exits
```

---

### 13.3 Exit Head Architectures — Full Specification

**Local Perception Head (LPH) — for Exits 1 and 2 (CNN stages):**

```python
class LocalPerceptionHead(nn.Module):
    def __init__(self, in_channels, num_classes, spatial_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(4)          # reduce spatial to 4×4
        self.conv = nn.Conv2d(in_channels, in_channels, 
                              kernel_size=3, padding=1, 
                              groups=in_channels)     # depthwise conv
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.flat_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.relu(self.bn(self.conv(x)))
        x = self.flat_pool(x).flatten(1)
        return self.fc(x)
```

**Global Aggregation Head (GAH) — for Exits 3, 4, and 5 (transformer stages):**

```python
class GlobalAggregationHead(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim=128):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)          # global spatial average
        self.proj = nn.Linear(in_channels, hidden_dim)
        self.act = nn.GELU()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (B, C, H, W) from MobileViT block output
        x = self.pool(x).flatten(1)                  # (B, C)
        x = self.norm(x)
        x = self.act(self.proj(x))
        return self.fc(x)
```

**Exit placement — exact layer hooks in MobileViT-S:**

| Exit | After timm layer | in_channels | Head type |
|---|---|---|---|
| Exit 1 | `stages.0` (after stage 1 MV2 blocks) | 32 | LPH |
| Exit 2 | `stages.1` (after stage 2 MV2 blocks) | 64 | LPH |
| Exit 3 | `stages.2` (after MobileViT block, dim=96) | 96 | GAH |
| Exit 4 | `stages.3` (after MobileViT block, dim=120) | 128 | GAH |
| Exit 5 | `stages.4` (after MobileViT block, dim=144) | 160 | GAH |

Verify these channel dimensions by running:
```python
import timm, torch
m = timm.create_model('mobilevit_s', pretrained=False, features_only=True)
out = m(torch.randn(1, 3, 256, 256))
for i, o in enumerate(out): print(f"Stage {i}: {o.shape}")
```
Use the printed channel counts if they differ from the table above.

---

### 13.4 Training Hyperparameters — All Stages

**Stage 0 (Backbone CIFAR adaptation):**
```yaml
optimizer: AdamW
lr: 1.0e-4
weight_decay: 1.0e-4
scheduler: CosineAnnealingLR
T_max: 10
epochs: 10
batch_size: 128
loss: CrossEntropyLoss
```

**Stage 1 (Backbone + exits, joint supervision):**
```yaml
optimizer: AdamW
lr: 5.0e-5          # lower than Stage 0 — backbone is already adapted
weight_decay: 1.0e-4
scheduler: CosineAnnealingLR
T_max: 40
epochs: 40
batch_size: 128
loss: weighted_sum_cross_entropy
exit_loss_weights: [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]  # exits 1–5 + final
# Lower weights on early exits prevent backbone degradation
# Final layer always weight 1.0
```

**Stage 2 (Exit distillation, backbone frozen):**
```yaml
freeze_backbone: true
optimizer: AdamW
lr: 1.0e-3          # higher — only exit heads are trained
weight_decay: 1.0e-4
scheduler: CosineAnnealingLR
T_max: 30
epochs: 30
batch_size: 128
loss: kl_divergence_plus_ce
kd_temperature: 4.0      # soften teacher logits
kd_alpha: 0.7            # weight of KD loss vs hard CE loss
# total_loss = kd_alpha * KL(soft) + (1 - kd_alpha) * CE(hard)
teacher: final_exit      # logits from exit 5 (final layer)
```

---

### 13.5 Confidence Threshold Setting

**Method:** Percentile-based calibration on the validation set. Set per-exit threshold such that X% of clean validation samples exit at or before that exit under normal (no attack) conditions.

**Default calibration target:** Each exit should handle approximately 20% of samples under clean conditions (uniform distribution across 5 exits). In practice, early exits handle easy samples — this will not be perfectly uniform, but use it as the optimization target.

**Threshold calibration procedure:**
```python
def calibrate_thresholds(model, val_loader, target_exit_rate=0.20):
    """
    For each exit i, find threshold t_i such that approximately
    target_exit_rate of validation samples have max_prob >= t_i
    at exit i (given they haven't exited earlier).
    Returns list of 5 thresholds.
    """
    # Collect confidence scores at each exit across val set
    # Binary search over threshold in [0.5, 0.99] for each exit
    # Return thresholds as list: [t1, t2, t3, t4, t5]
```

**Default thresholds (use if calibration is not yet run):**
```yaml
exit_thresholds: [0.70, 0.80, 0.85, 0.88, 0.90]
# Conservative — err toward letting samples propagate further
# Calibrate on val set before running attack experiments
```

**Threshold format in config:**
```yaml
# configs/mobilevit_s.yaml
thresholds:
  exit1: 0.70
  exit2: 0.80
  exit3: 0.85
  exit4: 0.88
  exit5: 0.90   # rarely used — almost all samples exit before this
  calibrate_from_val: true   # if true, overrides above with calibrated values
```

---

### 13.6 Entropy Computation

```python
import torch
import torch.nn.functional as F

def prediction_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Shannon entropy of softmax distribution.
    logits: (B, num_classes)
    returns: (B,) entropy values in nats
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def max_confidence(logits: torch.Tensor) -> torch.Tensor:
    """
    Maximum softmax probability (used for exit decisions).
    logits: (B, num_classes)
    returns: (B,) max prob values
    """
    probs = F.softmax(logits, dim=-1)
    return probs.max(dim=-1).values

def exit_decision(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Returns True for samples that should exit (confident enough).
    logits: (B, num_classes)
    threshold: float
    returns: (B,) boolean mask
    """
    return max_confidence(logits) >= threshold
```

**Entropy normalization:** Normalize by `log(num_classes)` to get values in [0, 1]. Use normalized entropy in loss functions, raw entropy for logging.

---

### 13.7 Attack Hyperparameters — HybridDDAS

```yaml
# configs/attack_config.yaml

attack:
  epsilon: 0.03137  # 8/255 — L-inf perturbation budget
  alpha: 0.00392    # 1/255 — PGD step size
  iterations: 20    # T — default; sweep {10, 20, 50} for ablation
  random_init: true # Initialize δ from Uniform(-ε, ε)

  # Loss coefficients
  lambda_perturb: 0.01   # L2 penalty on perturbation magnitude
  mu_pred: 10.0          # Penalty for changing final prediction
                         # High value — strongly preserve final label

  # Stage-aware loss weights (Experiment A = uniform default)
  exit_entropy_weights:
    experiment_A: [1.0, 1.0, 1.0, 1.0, 1.0]   # uniform
    experiment_B: [2.0, 2.0, 1.0, 0.5, 0.5]   # CNN-first
    experiment_C: [0.5, 0.5, 1.0, 2.0, 2.0]   # transformer-first
  default_experiment: A

  # Prediction preservation
  preserve_prediction: true
  # Method: penalize if argmax of final exit changes from clean prediction
  # pred_change_penalty = mu_pred * (pred_changed.float())
```

**Full loss function — explicit:**

```python
def hybrid_ddas_loss(clean_logits_list, adv_logits_list, delta, 
                     exit_weights, lambda_perturb, mu_pred):
    """
    clean_logits_list: list of 5 tensors (B, C) from clean forward pass
    adv_logits_list:   list of 5 tensors (B, C) from adversarial forward pass
    delta:             (B, 3, H, W) perturbation
    exit_weights:      list of 5 floats
    """
    # 1. Entropy maximization at each exit
    entropy_loss = 0
    for i, (logits, w) in enumerate(zip(adv_logits_list, exit_weights)):
        ent = prediction_entropy(logits)              # (B,)
        entropy_loss += w * (-ent.mean())             # maximize = minimize negative

    # 2. Perturbation magnitude penalty
    perturb_loss = lambda_perturb * delta.norm(p=2, dim=(1,2,3)).mean()

    # 3. Prediction preservation penalty (on final exit only)
    clean_pred = clean_logits_list[-1].argmax(dim=-1)   # (B,)
    adv_pred   = adv_logits_list[-1].argmax(dim=-1)     # (B,)
    pred_changed = (clean_pred != adv_pred).float()
    pred_loss = mu_pred * pred_changed.mean()

    return entropy_loss + perturb_loss + pred_loss
```

---

### 13.8 Baseline Attack Implementations

Both baselines target prediction accuracy (not resource exhaustion) — they are included for comparison, showing that standard attacks do NOT produce the same resource impact as HybridDDAS.

**ILFO (Input-dependent Layer-Fading Output):**
- Source: Heidari & Shehu, "ILFO: Adversarial Attack on Adaptive Neural Networks," NeurIPS 2020 Workshop
- Mechanism: Maximizes the sum of L2 norms of intermediate feature maps at each exit, pushing the model to rely on deeper layers
- Implementation:

```python
def ilfo_attack(model, x, epsilon, alpha, iterations):
    """
    Maximize intermediate feature map norms to force deep computation.
    """
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-epsilon, epsilon)
    x_adv = x_adv.clamp(0, 1)
    
    for _ in range(iterations):
        x_adv.requires_grad_(True)
        features = model.get_intermediate_features(x_adv)  # list of feature maps
        loss = -sum(f.norm(p=2, dim=(1,2,3)).mean() for f in features)
        loss.backward()
        x_adv = x_adv.detach() + alpha * x_adv.grad.sign()
        delta = (x_adv - x).clamp(-epsilon, epsilon)
        x_adv = (x + delta).clamp(0, 1).detach()
    
    return x_adv
```

**DeepSloth:**
- Source: Hong et al., "DeepSloth: A Universal Adversarial Perturbation That Can Slow Down Neural Network Inference," CVPR 2022 Workshop
- Mechanism: Universal perturbation (not input-specific) that minimizes early-exit confidence across the dataset
- Implementation: Compute universal δ by aggregating gradients across a subset of training samples; apply same δ to all test inputs
- Key difference from HybridDDAS: universal (not adaptive per-input), targets confidence directly (not entropy)

```python
def deepsloth_attack(model, train_loader, epsilon, alpha, 
                     iterations, n_samples=500):
    """
    Learn universal perturbation that suppresses early exit confidence.
    """
    delta = torch.zeros(1, 3, 256, 256).cuda()
    
    for _ in range(iterations):
        grad_accum = torch.zeros_like(delta)
        count = 0
        for x, _ in train_loader:
            if count >= n_samples: break
            x = x.cuda()
            x_adv = (x + delta).clamp(0, 1)
            x_adv.requires_grad_(True)
            exit_logits = model.get_all_exit_logits(x_adv)
            # Minimize max confidence at each exit
            loss = sum(max_confidence(l).mean() for l in exit_logits[:-1])
            loss.backward()
            grad_accum += x_adv.grad.sum(0, keepdim=True)
            count += len(x)
        
        delta = delta - alpha * grad_accum.sign()
        delta = delta.clamp(-epsilon, epsilon)
    
    return delta  # universal perturbation — add to any input
```

---

### 13.9 Metrics — Exact Computation

**Attack Persistence Rate (APR):**
```python
def compute_apr(model, x_adv, thresholds):
    """
    Fraction of adversarial samples that pass ALL early exits
    (reach the final layer without exiting early).
    """
    n = len(x_adv)
    reached_final = 0
    with torch.no_grad():
        for x in x_adv:
            exited_early = False
            exit_logits = model.get_all_exit_logits(x.unsqueeze(0))
            for i, (logits, thresh) in enumerate(zip(exit_logits[:-1], thresholds)):
                if max_confidence(logits).item() >= thresh:
                    exited_early = True
                    break
            if not exited_early:
                reached_final += 1
    return reached_final / n
```

**Stage-level APR (novel metric):**
```python
def compute_stage_apr(model, x_adv, thresholds, stage_type):
    """
    APR computed only over exits of a given type.
    stage_type: 'cnn' (exits 1-2) or 'transformer' (exits 3-5)
    
    Definition: among samples that reach a given exit,
    what fraction does NOT exit there (i.e., attack persists past it)?
    """
    exit_indices = [0, 1] if stage_type == 'cnn' else [2, 3, 4]
    # ... compute per-exit persistence rates for the specified indices
```

**PSNR:**
```python
def compute_psnr(x_clean, x_adv):
    mse = F.mse_loss(x_adv, x_clean)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
```

---

### 13.10 Results Table Schema

The agent should produce these exact tables as CSV files in `results/` for each dataset:

**Table 1 — Main results:**
```
model | dataset | attack | APR (%) | latency_increase (%) | power_increase (%) | accuracy_clean (%) | accuracy_adv (%) | PSNR (dB)
```

**Table 2 — Stage-level APR:**
```
model | dataset | attack | APR_cnn_exits (%) | APR_transformer_exits (%) | APR_overall (%)
```

**Table 3 — Ablation (stage weighting):**
```
experiment | exit_weights | APR (%) | PSNR (dB) | latency_increase (%)
```

**Table 4 — Perturbation budget sweep:**
```
epsilon | APR (%) | PSNR (dB) | accuracy_adv (%)
```

All tables: report mean ± std over 3 random seeds. Save raw results as JSON before computing statistics.

---

### 13.11 Agentic Implementation Scope — Phase 1

Implement in this order. Stop and report results after each phase before proceeding.

**Phase 1 (scaffolding + exit grafting):**
- [ ] Repository structure from Section 9.1
- [ ] `models/exit_heads.py` — LPH and GAH (Section 13.3)
- [ ] `models/mobilevit_s_exits.py` — backbone loading + exit grafting (Sections 2.1, 13.2, 13.3)
- [ ] `attacks/utils/entropy.py` — entropy, confidence, exit decision functions (Section 13.6)
- [ ] `training/train_exits_stage0.py` — backbone CIFAR adaptation (Section 13.4)
- [ ] `training/train_exits_stage1.py` — joint exit supervision (Section 13.4)
- [ ] `training/train_exits_stage2.py` — exit distillation (Section 13.4)
- [ ] `configs/mobilevit_s.yaml` — full config (Sections 13.3, 13.5)
- [ ] `eval/eval_clean.py` — accuracy per exit, threshold calibration (Section 13.5)

**Do not implement in Phase 1:**
- Attack code
- Baseline implementations
- Hardware profiling
- MobileNetV4 secondary target

**Phase 1 success gate:** Running `python eval/eval_clean.py` on CIFAR-10 after Stage 2 training produces per-exit accuracy meeting the thresholds in Section 3.3.