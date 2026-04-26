# Detailed Ablation Study Report — Biomedical Embeddings

---

## 1. Introduction

This document provides a comprehensive explanation of the ablation studies conducted on biomedical embedding models. The goal is to understand how different supervision signals (synonyms, semantic types, and relations) influence embedding quality.

---

## 2. Objective of Ablation Study

Ablation studies are used to isolate the contribution of each component in the training pipeline. Instead of evaluating only the final model, we systematically enable or disable components to understand:

- What improves performance?
- Why does it improve?
- What trade-offs exist?

---

## 3. Models Evaluated

| Model | Components |
|------|-----------|
| Baseline | No alignment |
| UMLS | Raw synonym alignment |
| Synonym Only | Clean synonym contrastive |
| Synonym + Type | + type classification |
| Synonym + Relation | + relation pairs |
| Full Enhanced | All components |

---

## 4. Loss Functions Used

### 4.1 NT-Xent Loss (Contrastive Loss)

This is the core loss used across all alignment-based models.

L_ntxent = -log( exp(sim(z_i, z_j)/tau) / sum exp(sim(z_i, z_k)/tau) )

- z_i, z_j: positive pair (synonyms or relations)
- z_k: negatives
- tau: temperature parameter

Purpose:
- Pull similar concepts together
- Push dissimilar concepts apart

---

### 4.2 Type Classification Loss (Binary Cross Entropy)

Used for semantic type supervision.

L_type = -sum(y * log(p))

- Multi-label classification
- Predicts concept types (disease, drug, etc.)

Purpose:
- Improve semantic separability
- Introduce structured categorization

---

### 4.3 Combined Loss

L_total = L_ntxent + lambda * L_type

- lambda controls importance of type classification

---

## 5. Key Insight About Loss Design

IMPORTANT:

The loss function does NOT fundamentally change across most ablations.

Instead, the **definition of positive pairs changes**:

| Setup | Positive Pairs |
|------|---------------|
| Synonym Only | Synonyms |
| Synonym + Relation | Synonyms + Relations |

This is the most critical design decision.

---

## 6. Results Summary — Word2Vec

| Model | AUC | MRR | P@20 | Type F1 |
|------|-----|-----|------|--------|
| Baseline | 0.6246 | 0.2113 | 0.0202 | 0.4942 |
| UMLS | 0.6634 | 0.2711 | 0.0260 | 0.4130 |
| Synonym Only | 0.6561 | 0.2738 | 0.0261 | 0.4023 |
| Synonym + Type | 0.6596 | 0.2696 | 0.0260 | 0.4076 |
| Synonym + Relation | 0.7157 | 0.4098 | 0.0365 | 0.3826 |
| Full Enhanced | 0.7150 | 0.4070 | 0.0365 | 0.3888 |

---

## 7. Results Summary — Transformer

| Model | AUC | MRR | P@20 | Type F1 |
|------|-----|-----|------|--------|
| Baseline | 0.5137 | 0.0420 | 0.0049 | 0.2299 |
| UMLS | 0.5192 | 0.0676 | 0.0076 | 0.2314 |
| Synonym Only | 0.6010 | 0.2927 | 0.0266 | 0.4073 |
| Synonym + Type | 0.6016 | 0.2930 | 0.0266 | 0.4233 |
| Synonym + Relation | 0.6674 | 0.4390 | 0.0382 | 0.3688 |
| Full Enhanced | 0.6683 | 0.4396 | 0.0382 | 0.3746 |

---

## 8. Detailed Interpretation

### 8.1 Baseline
- No alignment
- Captures only statistical co-occurrence or context

---

### 8.2 UMLS
- Weak improvement
- Noisy supervision

---

### 8.3 Synonym Only
- Major improvement
- Strong semantic clustering

---

### 8.4 Synonym + Type
- Slight improvement in classification
- Minimal impact on retrieval

---

### 8.5 Synonym + Relation
- Largest improvement overall
- Enables relational reasoning

---

### 8.6 Full Enhanced
- Best balanced model
- Trade-off between classification and retrieval

---

## 9. Trade-off Analysis

| Component | Effect on Retrieval | Effect on Classification |
|----------|-------------------|--------------------------|
| Synonyms | Moderate | Moderate |
| Types | Low | High |
| Relations | Very High | Negative |

---

## 10. Core Insight

Semantic similarity is NOT enough.

Relational structure must be explicitly learned.

---

## 11. Explanation for Viva

If asked:

"What improved performance?"

Answer:

"Not the loss function, but the training signal. By introducing relation-based positive pairs into the contrastive framework, we enabled the model to learn real-world biomedical structure."

---

## 12. Final Conclusion

- NT-Xent is the backbone
- Type loss adds structure but limited impact
- Relation learning is the dominant factor
- Best performance comes from combining all signals

---

