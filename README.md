# RoViT-Detect
Multimodal Social Media Fake News Detection Using RoBERTa and Vision Transformer Encoders with Reliability Aware Adaptive Fusion

# RoViT-Detect — Code Explanation & Deep Dive

**Paper:** Multimodal Social Media Fake News Detection Using RoBERTa and Vision
Transformer Encoders with Reliability Aware Adaptive Fusion  
**Code:** `rovit_detect_complete.py`  
**Platform:** Kaggle (GPU T4 × 2 recommended)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Design](#2-architecture-design)
3. [Dataset Classes](#3-dataset-classes)
4. [Fusion Modules](#4-fusion-modules)
5. [Unified Model — RoViTClassifier](#5-unified-model--rovitclassifier)
6. [Training Pipeline](#6-training-pipeline)
7. [Evaluation & Metrics](#7-evaluation--metrics)
8. [Main Experiments](#8-main-experiments)
9. [Ablation Studies](#9-ablation-studies)
10. [How to Run on Kaggle](#10-how-to-run-on-kaggle)
11. [Expected Results](#11-expected-results)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Project Overview

RoViT-Detect is a **two-stream late-fusion** multimodal classifier for fake news detection.

### Core Hypothesis — "Strong Backbone Hypothesis"

> When unimodal encoders are sufficiently pre-trained, their CLS representations are
> already semantically aligned, making complex cross-modal attention **redundant**.

The hypothesis is **empirically validated** through four ablation studies. The key finding:
Cross-Attention fusion achieves 99.72% vs Concat's 99.82% — a gap of only **Δ = 0.001**
despite adding ~2.1M extra parameters and requiring 7 epochs vs 4 to converge.

### Datasets

| Dataset  | Language | Samples  | Split type         | Key challenge         |
|----------|----------|----------|--------------------|----------------------|
| Twitter  | English  | ~16,340  | MediaEval event-level | Short noisy tweets  |
| Weibo    | Chinese  | ~9,528   | Event-level (pickle IDs) | Cross-lingual generalization |
| Fakeddit | English  | 800k+    | Reddit post-level  | Cross-domain diversity      |

All splits are **event-level** — no event appears in both training and testing, preventing
temporal/propagation data leakage.

---

## 2. Architecture Design

```
Input text  →  Tokenizer  →  [CLS, t1, t2, ..., tL, SEP]  (L_max=128)
                             ↓
                         RoBERTa (all layers fine-tuned)
                             ↓
                   CLS hidden state  ξ_T ∈ R^{d_T}
                   (d_T = 1024 for RoBERTa-large,
                    d_T = 768  for Chinese-RoBERTa)

Input image →  Resize(224×224) → Normalize (ImageNet) → Patch embedding (16×16)
                             ↓
                          ViT-Base (all layers fine-tuned)
                             ↓
                   CLS hidden state  φ_ϑ ∈ R^{768}

                ── FUSION ──
                Concat:      γ_ν = [ξ_T ; φ_ϑ]  ∈ R^{d_T+768}
                Cross-Attn:  bidirectional multi-head attention
                Co-Attn:     affinity-gated simultaneous weighting
                Gated:       learned per-sample modality weights

                ── MLP HEAD ──
                γ_ν  →  Linear(→512) + ReLU  →  Dropout(0.3)  →  Linear(→2)
                                                                     ↓
                                                              logits ∈ R^2
                                                       Softmax → p(real), p(fake)
```

**Loss function:** Cross-Entropy over 2 classes.

**Why CLS tokens?**
The `[CLS]` token in both RoBERTa and ViT is specially designed to aggregate
global sequence/image-level information. By epoch-end of pre-training, it represents
a holistic summary of the entire input, making it the natural choice for classification.

---

## 3. Dataset Classes

### `MultiModalDataset`

```python
dataset = MultiModalDataset(texts, image_paths, labels, tokenizer, transform, max_len)
```

Each `__getitem__` call:
1. **Tokenizes** the text string into `[input_ids, attention_mask]` tensors of shape `(max_len,)`
2. **Loads and transforms** the image: `PIL.open → convert('RGB') → Resize(224) → ToTensor → Normalize`
3. **Injects noise** (only for Ablation D) based on `.noise_type` and `.noise_level`
4. Returns a dict with keys: `input_ids`, `attention_mask`, `image`, `label`

**Fallback for missing images:**
```python
try:
    image = Image.open(path).convert("RGB")
    image = transform(image)
except Exception:
    image = torch.zeros(3, 224, 224)   # black image — graceful degradation
```

This is critical for Fakeddit (images not in our Kaggle distribution) — the model
still runs in text-dominant mode and achieves 85.14% accuracy.

### `make_loaders`

Creates `DataLoader` objects for train/val/test splits. Key settings:
- `pin_memory=True` for GPU (speeds up CPU→GPU transfer)
- `pin_memory=False` for TPU (TPU uses its own memory system)
- `num_workers=2` for parallel data loading

---

## 4. Fusion Modules

### 4.1 `CrossAttentionFusion`

```
text_proj(ξ_T) → query t ∈ R^{B×1×512}
vis_proj(φ_ϑ)  → key/value v ∈ R^{B×1×512}

t → v:  t2v_attn(query=t, key=v, value=v)  →  LayerNorm(t + output) → tv
v → t:  v2t_attn(query=v, key=t, value=t)  →  LayerNorm(v + output) → vt

output = Linear([tv ; vt], 512)
```

**Why 8 heads?** Each head learns to focus on a different aspect of the cross-modal
relationship (e.g., sentiment vs. visual sentiment, topic vs. visual theme).

**Residual connections** (`t + tv`) prevent the original information from being lost
if the attention mechanism finds nothing useful to add.

### 4.2 `CoAttentionFusion`

```
ht = tanh(tp(ξ_T))        ← projected text representation
hv = tanh(vp(φ_ϑ))        ← projected vision representation
aff = tanh(Bilinear(ht, hv))  ← joint affinity signal

w_t = sigmoid(Linear(ht + aff))  ← scalar gate for text (B×1)
w_v = sigmoid(Linear(hv + aff))  ← scalar gate for vision (B×1)

output = Linear([w_t * ht ; w_v * hv], 512)
```

The key difference from Cross-Attention: both modalities are gated **simultaneously**
using a shared affinity signal, rather than sequentially attending to each other.

### 4.3 `GatedFusion`

```
gate = Softmax(Linear(ReLU(Linear([ξ_T ; φ_ϑ])), 2))  ← (B, 2), sums to 1.0
g_t, g_v = gate[:, 0], gate[:, 1]

output = Linear(g_t * tp(ξ_T) + g_v * vp(φ_ϑ), 512)
```

This is the closest approximation to "reliability-aware" fusion — the gate network
assigns per-sample importance weights based on the joint input. Despite this explicit
reliability modeling, Ablation B shows it achieves 99.51% vs Concat's 99.82%.

**Interpretation:** The pre-trained encoders already produce representations that
implicitly encode reliability — no additional gating mechanism adds meaningful signal.

---

## 5. Unified Model — `RoViTClassifier`

A single class handles all 6 fusion variants, controlled by `fusion_type`:

```python
model = RoViTClassifier(
    n_classes        = 2,
    text_model_name  = "roberta-large",
    vision_model_name= "vit_base_patch16_224.augreg_in21k",
    fusion_type      = "concat",     # or "cross_attn", "co_attn", "gated",
                                     # "text_only", "image_only"
    freeze_text      = False,        # True for Ablation C
    freeze_vision    = False,        # True for Ablation C
    dropout          = 0.3,
)
```

**Parameter counts (approximate):**

| Fusion type  | Extra params | Total params |
|--------------|-------------|--------------|
| text_only    | 0           | ~355M        |
| image_only   | 0           | ~86M         |
| concat       | 0           | ~441M        |
| cross_attn   | +2.1M       | ~443M        |
| co_attn      | +2.6M       | ~444M        |
| gated        | +1.6M       | ~443M        |

**Freezing behavior:**
When `freeze_text=True`, all RoBERTa parameters get `requires_grad=False`.
This means AdamW will not update them — they act as fixed feature extractors.
The Ablation C results show this costs ~3.3% accuracy, proving fine-tuning matters.

---

## 6. Training Pipeline

### 6.1 Optimizer — AdamW

```
lr           = 2×10⁻⁵   (standard for transformer fine-tuning)
weight_decay = 0.01      (L2 regularization — prevents overfitting)
```

**Why AdamW over Adam?**
Adam's L2 decay interacts incorrectly with adaptive learning rates. AdamW decouples
weight decay from gradient updates, giving better regularization for large transformers.

### 6.2 Learning Rate Schedule

```
warmup → linear decay
       ├─ warmup:  steps 0 → 10% of total   LR: 0 → 2×10⁻⁵
       └─ decay:  steps 10% → 100% of total  LR: 2×10⁻⁵ → 0
```

**Why warmup?** Large transformers have unstable gradients in the first few batches.
Starting with LR=0 and gradually increasing prevents the first few batches from
causing large weight updates that disrupt the pre-trained representations.

### 6.3 Gradient Clipping

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Clips all gradients to have L2 norm ≤ 1.0. Prevents exploding gradients which
are common when fine-tuning large models on small datasets.

### 6.4 Early Stopping

```python
early_stopper = EarlyStopping(patience=3, min_delta=0.001)
```

Training stops when validation accuracy does not improve by at least 0.001 for
3 consecutive epochs. This prevents overfitting (observable from epoch 4+ in Twitter)
while ensuring convergence is real (not noise).

### 6.5 GPU Memory Management

After each `run_experiment()` call:
```python
del model, optimizer, scheduler, loss_fn
clear_vram()   # gc.collect() + torch.cuda.empty_cache()
```

This is **critical** for Ablation B/C/D where multiple models are trained sequentially.
Without explicit memory clearing, VRAM accumulates and causes OOM errors by the 3rd model.
The T4 GPU has 14.5 GB — each model occupies ~13-14 GB during training (model weights +
optimizer states = ~3× model size).

---

## 7. Evaluation & Metrics

### `evaluate()` function

Returns a dict with:
- `labels`: ground truth (numpy array)
- `preds`: argmax predictions (numpy array)
- `probs`: softmax probability for class 1 / "Fake" (numpy array)
- `accuracy`: scalar
- `loss`: average CE loss

### `compute_all_metrics()` — 6 metrics

| Metric        | Why it matters                                          |
|---------------|---------------------------------------------------------|
| `accuracy`    | Overall correct predictions / total                    |
| `precision`   | Of predicted fakes, how many are actually fake?        |
| `recall`      | Of actual fakes, how many did we catch?                |
| `f1_weighted` | Harmonic mean weighted by class support                |
| `f1_macro`    | Harmonic mean treating all classes equally             |
| `auc`         | Discriminative ability across all decision thresholds  |

**Why AUC matters for fake news detection:**
AUC = P(model scores fake sample higher than real sample). An AUC of 1.0 means
perfect discriminative ability regardless of threshold, even with class imbalance.
The Twitter AUC = 1.0000 confirms the model's near-perfect class separation.

---

## 8. Main Experiments

### Experiment 1: Twitter (English)

```
Backbone: RoBERTa-large (1024-d CLS) + ViT-Base (768-d CLS)
Fusion:   Concatenation → Linear(1792→512) + ReLU + Dropout(0.3) + Linear(512→2)
Result:   Acc=99.69%  Precision=99.70%  Recall=99.69%  F1=99.69%  AUC=1.0000
Epochs:   6 (early stopping at epoch 6, best at epoch 3)
```

The near-immediate convergence (95%+ by epoch 1) reflects the quality of
pre-trained representations. The model does not need to learn fake news patterns
from scratch — it adapts its existing knowledge of language and vision.

### Experiment 2: Weibo (Chinese)

```
Backbone: hfl/chinese-roberta-wwm-ext (768-d CLS) + ViT-Base (768-d CLS)
Note:     Architecture identical to Twitter — ONLY text encoder swapped
Result:   Acc=98.93%  Precision=99.0%  Recall=98.9%  F1=98.93%  AUC=0.9971
Epochs:   7 (early stopping at epoch 7, best at epoch 4)
```

The asymmetric precision/recall (Real: P=0.999, R=0.979 vs Fake: P=0.980, R=0.999)
indicates a slight conservative bias — the model prefers to flag ambiguous content
as fake rather than miss fake news.

### Experiment 3: Fakeddit (English, text-dominant)

```
Backbone: RoBERTa-base (768-d CLS) + ViT-Base (zeros, no images available)
Result:   Acc=85.14%  Precision=85.5%  Recall=85.0%  F1=85.2%  AUC=0.9262
Epochs:   6 (early stopping at epoch 6)
```

The ViT receives a constant zero-tensor (black image). Performance is entirely
driven by the text encoder. The 85.14% accuracy on a 6-category cross-domain
dataset demonstrates strong RoBERTa generalization.

---

## 9. Ablation Studies

### Ablation A: Modality Contribution

**Setup:** 3 variants trained independently. Twitter test set.

```
Text-only  (RoBERTa-large, no ViT): Acc=92.75%  AUC=0.9775
Image-only (ViT-Base, no text):     Acc=99.54%  AUC=0.9999
Multimodal (Concat, both):          Acc=99.88%  AUC=1.0000
```

**Key finding:** Image-only (99.54%) outperforms text-only (92.75%) by 6.79%.
Twitter posts have highly distinctive visual patterns — manipulated or out-of-context
images are extremely discriminative for ViT. Fusion adds a further 0.34%.

**Implementation note:** `fusion_type="text_only"` loads only RoBERTa and still
accepts an `image` tensor in `forward()` — it simply ignores it. This unified
interface allows the same training loop to work for all 3 variants.

### Ablation B: Fusion Strategy

**Setup:** Identical RoBERTa-large + ViT-Base. Only the fusion module changes.
Batch size reduced to 8 for co_attn and gated (BiLinear / gate network use more VRAM).

```
Twitter:
  Concat      (bs=16, 4 epochs):  Acc=99.82%  AUC=0.9999
  Cross-Attn  (bs=16, 7 epochs):  Acc=99.72%  AUC=1.0000  [+7 epochs, +2.1M params]
  Co-Attn     (bs= 8, 6 epochs):  Acc=99.72%  AUC=1.0000
  Gated       (bs= 8, 4 epochs):  Acc=99.51%  AUC=1.0000

Weibo:
  Concat      : Acc=97.80%  AUC=0.9962
  Cross-Attn  : Acc=98.02%  AUC=0.9968
  Co-Attn     : Acc=97.63%  AUC=0.9955
  Gated       : Acc=97.48%  AUC=0.9949
```

**The critical finding:** All four strategies achieve AUC = 1.0000 on Twitter,
with accuracy differences of ≤0.003. Complex attention adds **zero meaningful
discriminative benefit** when pre-trained encoders produce high-quality representations.

### Ablation C: Training Strategy

**Setup:** Concat fusion only. Vary frozen/unfrozen backbone.

```
Full fine-tune (both trainable):   Acc=99.82%  (baseline)
Frozen ViT (only RoBERTa trains):  Acc=97.68%  (-2.14%)
Frozen RoBERTa (only ViT trains):  Acc=96.54%  (-3.28%)
Both frozen (feature extraction):  Acc=88.74%  (-11.08%)
```

**Interpretation:** Fine-tuning is essential. Pre-trained features alone (Both frozen)
provide a reasonable 88.74% baseline, but adapting the representations to the specific
distribution of fake news content is critical for peak performance.

**Why Frozen ViT > Frozen RoBERTa?**
Consistent with Ablation A: the ViT pre-trained features are more discriminative
for this task (99.54% image-only vs 92.75% text-only). When ViT is fine-tuned,
it adapts its visual patches to focus on manipulation artifacts and contextual
inconsistencies — patterns not present in its JFT-300M pre-training distribution.

### Ablation D: Noisy Modality Robustness

**Setup:** Best Twitter concat model evaluated on test set with 7 noise conditions.
No retraining — evaluates robustness of learned representations.

```
No noise (baseline):     Acc=99.82%  AUC=0.9999
Gaussian img  σ=0.5:     Acc=99.34%  AUC=0.9989  (-0.48%)
Gaussian img  σ=1.0:     Acc=98.71%  AUC=0.9972  (-1.11%)
Zero (black) image:      Acc=93.87%  AUC=0.9819  (-5.95%)  ← approaches text-only baseline
Random noise image:      Acc=93.51%  AUC=0.9802  (-6.31%)
Text masked 30%:         Acc=97.89%  AUC=0.9962  (-1.93%)
Text masked 50%:         Acc=96.23%  AUC=0.9921  (-3.59%)
```

**Key insight — modality dominance analysis:**
- Zero/random image → 93-94% (approaches text-only 92.75%): visual stream removed
- Text masked 50% → 96.23% (image stream compensates): text stream degraded

Neither modality is completely dominant. The model degrades **gracefully** rather than
catastrophically, demonstrating robust multimodal fusion. The MLP head has implicitly
learned to rely on the other modality when one produces uninformative features.

---

## 10. How to Run on Kaggle

### Step 1: Setup
1. Create a new notebook on kaggle.com
2. Settings → Accelerator → **GPU T4 × 2**
3. Copy-paste `rovit_detect_complete.py` as the notebook script
   (or upload as a `.py` file and `!python rovit_detect_complete.py`)

### Step 2: Add Datasets
Click **+ Add Data** and attach:
- `ayushshukla07/twitter-zip` → Twitter dataset
- `ayushshukla07/weibo-zip` → Weibo dataset
- `ayushshukla07/fakeddit` → Fakeddit dataset (optional)

### Step 3: Verify Paths
Check that the paths in Cell 4 match your dataset slugs:
```python
TWITTER_BASE  = "/kaggle/input/datasets/ayushshukla07/twitter-zip/twitter"
WEIBO_BASE    = "/kaggle/input/datasets/ayushshukla07/weibo-zip/weibo"
FAKEDDIT_BASE = "/kaggle/input/datasets/ayushshukla07/fakeddit"
```

### Step 4: Run All
The notebook runs sequentially: Experiments 1→2→3 → Ablations A→B→C→D → Summary.

**Total runtime estimate (GPU T4 × 2):**

| Section | Approx. time |
|---------|-------------|
| Experiment 1 (Twitter)  | ~25 min |
| Experiment 2 (Weibo)    | ~20 min |
| Experiment 3 (Fakeddit) | ~30 min |
| Ablation A (3 variants) | ~45 min |
| Ablation B (4+4 variants)| ~120 min |
| Ablation C (4 variants) | ~60 min |
| Ablation D (eval only)  | ~5 min  |
| **Total**               | **~5 hours** |

**Tip:** If GPU quota is limited, run Experiments 1-3 first, then Ablations in a
second session. The best model `.pth` files are saved to `/kaggle/working/` and
can be attached as a dataset for the second session.

---

## 11. Expected Results

### Main Experiments

| Dataset  | Accuracy | Precision | Recall  | F1-Macro | AUC    |
|----------|----------|-----------|---------|----------|--------|
| Twitter  | 99.69%   | 99.70%    | 99.69%  | 99.69%   | 1.0000 |
| Weibo    | 98.93%   | 99.0%     | 98.9%   | 98.93%   | 0.9971 |
| Fakeddit | 85.14%   | 85.5%     | 85.0%   | 85.2%    | 0.9262 |

### Ablation A — Modality (Twitter)

| Variant       | Accuracy | F1-Macro | AUC    |
|---------------|----------|----------|--------|
| Text-only     | 92.75%   | 0.9275   | 0.9775 |
| Image-only    | 99.54%   | 0.9954   | 0.9999 |
| **Multimodal**| **99.88%** | **0.9988** | **1.0000** |

### Ablation B — Fusion (Twitter)

| Strategy    | Accuracy | AUC    | Extra params | Epochs |
|-------------|----------|--------|-------------|--------|
| **Concat**  | **99.82%** | 0.9999 | 0          | 4      |
| Cross-Attn  | 99.72%   | 1.0000 | +2.1M      | 7      |
| Co-Attn     | 99.72%   | 1.0000 | +2.6M      | 6      |
| Gated       | 99.51%   | 1.0000 | +1.6M      | 4      |

### Ablation C — Training Strategy (Twitter)

| Strategy        | Accuracy | Drop vs Full |
|-----------------|----------|-------------|
| Full fine-tune  | 99.82%   | —           |
| Frozen ViT      | 97.68%   | -2.14%      |
| Frozen RoBERTa  | 96.54%   | -3.28%      |
| Both frozen     | 88.74%   | -11.08%     |

---

## 12. Troubleshooting

### `OutOfMemoryError: CUDA out of memory`
**Cause:** Previous model still in VRAM when next model loads.  
**Fix:** The `clear_vram()` call after each `run_experiment()` should handle this.
If it still occurs: reduce `batch_size` in `HP_SMALL` from 8 to 4.

### `FileNotFoundError: No .tsv files under FAKEDDIT_BASE`
**Cause:** Fakeddit dataset not attached or path mismatch.  
**Fix:** Check the dataset slug: `!ls /kaggle/input/datasets/ayushshukla07/` and
update `FAKEDDIT_BASE` to match.

### `NameError: name 'tw_trld' is not defined` (in Ablation B)
**Cause:** Running Ablation B in a new Kaggle session without running Cells 10-11.  
**Fix:** Run the entire notebook from Cell 1. The data loaders are created in Cells 10-11.

### `AttributeError: RobertaTokenizer has no attribute encode_plus`
**Cause:** transformers ≥ 4.40 removed `encode_plus`.  
**Fix:** The code uses `self.tokenizer(...)` directly (the `__call__` method), which
is the modern API. If you see this error, ensure `transformers` is updated:
`!pip install --upgrade transformers`

### Slow training despite 2x T4
**Cause:** `num_workers=2` may be low for the CPU on Kaggle.  
**Fix:** Try `num_workers=4` in `make_loaders()`.

### Warning: `403 Forbidden: Discussions are disabled for this repo`
**Cause:** HuggingFace background thread trying to convert `hfl/chinese-roberta-wwm-ext`
to safetensors format. This is a **harmless background warning** and does not affect
training. The model loads and trains correctly.

---

## Code Structure Summary

```
rovit_detect_complete.py
│
├── CELL 1  : pip install
├── CELL 2  : Accelerator detection (TPU / Multi-GPU / Single-GPU)
├── CELL 3  : Imports
├── CELL 4  : Global config (paths, model names, hyperparameters)
│
├── CELL 5  : Dataset utilities
│   ├── MultiModalDataset  — text+image+label with noise injection
│   ├── make_loaders       — creates DataLoader triplet
│   ├── _filter_images     — removes rows with missing images
│   └── print_dataset_stats
│
├── CELL 6  : Fusion modules
│   ├── CrossAttentionFusion  — bidirectional multi-head attention
│   ├── CoAttentionFusion     — affinity-gated co-attention
│   └── GatedFusion           — learnable per-sample reliability gates
│
├── CELL 7  : RoViTClassifier (unified model, all 6 fusion types)
│
├── CELL 8  : Training & evaluation
│   ├── EarlyStopping      — patience-based convergence criterion
│   ├── train_epoch        — one forward+backward pass over loader
│   ├── evaluate           — inference with metrics collection
│   ├── compute_all_metrics— computes 6 metrics from evaluate() output
│   ├── show_table         — pretty-prints result dicts
│   ├── run_experiment     — full train→select→evaluate pipeline
│   └── clear_vram         — VRAM cleanup between experiments
│
├── CELL 9  : Data loaders
│   ├── load_twitter()    — MediaEval event-level split
│   ├── load_weibo()      — Weibo event-level split
│   └── load_fakeddit()   — auto-detects TSV filenames, handles no-image mode
│
├── CELLS 10-12 : Main experiments (Twitter / Weibo / Fakeddit)
│
├── CELL 13 : Ablation A — Modality contribution
├── CELL 14 : Ablation B — Fusion strategy (Twitter + Weibo)
├── CELL 15 : Ablation C — Training strategy
├── CELL 16 : Ablation D — Noisy modality robustness
│
└── CELL 17 : Final results table + CSV export
```

---

*Generated as part of the RoViT-Detect Major Revision — Discover Computing, Springer Nature.*  
*Submission ID: b0ad0220-446b-47ad-92ff-6ce6e47ebe87*
