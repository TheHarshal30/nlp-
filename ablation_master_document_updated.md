# Detailed Ablation Study Report — Biomedical Embeddings (with Model Differences)

---

## 1. Introduction

This document explains the ablation studies and **clearly differentiates each model variant** in terms of supervision, loss, and effect on the embedding space.

---

## 2. Objective of Ablation Study

- Isolate contribution of each component
- Understand *why* metrics change
- Identify trade-offs between retrieval and classification

---

## 3. Loss Functions Used

### 3.1 NT-Xent (Contrastive Loss)

L_ntxent = -log( exp(sim(zi, zj)/tau) / sum exp(sim(zi, zk)/tau) )

- zi, zj: positive pairs
- zk: negatives
- tau: temperature

### 3.2 Type Loss (Binary Cross Entropy)

L_type = -sum(y * log(p))

### 3.3 Combined

L_total = L_ntxent + lambda * L_type

---

## 4. What Actually Changes Across Models

**Critical Insight:**
> The loss function remains mostly the same.  
> What changes is the definition of positive pairs.

---

## 5. Model Variants + Differences

### 5.1 Baseline
- No alignment
- No contrastive loss
- Only pretraining objective (Word2Vec / MLM)

👉 Learns:
- co-occurrence (Word2Vec)
- context (Transformer)

---

### 5.2 UMLS Alignment

**What it uses:**
- NT-Xent loss
- UMLS synonym pairs (raw extraction)

**Difference from Baseline:**
- Adds contrastive learning
- Introduces weak supervision

**Limitation:**
- noisy pairs
- inconsistent mapping

---

### 5.3 Synonym Only (Clean Alignment)

**What it uses:**
- NT-Xent loss
- cleaned synonym pairs

**Difference from UMLS:**
- better quality positives
- more consistent signal

**Effect:**
- strong semantic clustering

👉 Key difference:
UMLS = noisy supervision  
Synonym only = clean supervision

---

### 5.4 Synonym + Type

**What it uses:**
- NT-Xent (synonyms)
- BCE type classification

**Difference from Synonym Only:**
- adds classification constraint

**Effect:**
- improves type separability
- minimal change in retrieval

👉 Key idea:
Adds **categorical structure**, not relational structure

---

### 5.5 Synonym + Relation

**What it uses:**
- NT-Xent
- positive pairs = synonyms + relations

**Difference from Synonym Only:**
- expands definition of “similar”

Before:
- diabetes ↔ diabetes mellitus

Now:
- diabetes ↔ insulin
- diabetes ↔ hyperglycemia

**Effect:**
- massive improvement in MRR and AUC

👉 Key idea:
Model learns **relationships instead of just similarity**

---

### 5.6 Full Enhanced

**What it uses:**
- NT-Xent (synonyms + relations)
- BCE type loss

**Difference from Synonym + Relation:**
- adds classification regularization

**Effect:**
- slightly improves type F1
- similar retrieval performance

👉 Trade-off:
- relation learning pulls concepts across types
- type loss pushes them apart

---

## 6. Summary of Differences

| Model | Positive Pairs | Additional Loss | Main Effect |
|------|---------------|----------------|------------|
| Baseline | None | None | Raw embeddings |
| UMLS | Noisy synonyms | NT-Xent | Weak alignment |
| Synonym Only | Clean synonyms | NT-Xent | Strong clustering |
| Syn + Type | Synonyms | + BCE | Better classification |
| Syn + Relation | Syn + Relations | NT-Xent | Strong retrieval |
| Full | Syn + Relations | + BCE | Balanced model |

---

## 7. Results (Word2Vec)

| Model | AUC | MRR | P@20 | Type F1 |
|------|-----|-----|------|--------|
| Baseline | 0.6246 | 0.2113 | 0.0202 | 0.4942 |
| UMLS | 0.6634 | 0.2711 | 0.0260 | 0.4130 |
| Synonym Only | 0.6561 | 0.2738 | 0.0261 | 0.4023 |
| Syn + Type | 0.6596 | 0.2696 | 0.0260 | 0.4076 |
| Syn + Relation | 0.7157 | 0.4098 | 0.0365 | 0.3826 |
| Full | 0.7150 | 0.4070 | 0.0365 | 0.3888 |

---

## 8. Results (Transformer)

| Model | AUC | MRR | P@20 | Type F1 |
|------|-----|-----|------|--------|
| Baseline | 0.5137 | 0.0420 | 0.0049 | 0.2299 |
| UMLS | 0.5192 | 0.0676 | 0.0076 | 0.2314 |
| Synonym Only | 0.6010 | 0.2927 | 0.0266 | 0.4073 |
| Syn + Type | 0.6016 | 0.2930 | 0.0266 | 0.4233 |
| Syn + Relation | 0.6674 | 0.4390 | 0.0382 | 0.3688 |
| Full | 0.6683 | 0.4396 | 0.0382 | 0.3746 |

---

## 9. Final Insight

- Synonyms → similarity  
- Types → categorization  
- Relations → reasoning  

> The dominant improvement comes from redefining similarity to include real-world relationships.

---

## 10. Viva Explanation (Simple)

“If we only use synonyms, the model learns meaning.  
If we add types, it learns categories.  
If we add relations, it learns how the medical world is connected.”

---

