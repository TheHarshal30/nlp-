
# Biomedical Embedding Study — Projection Head, Ablation, and Architectures (Repo-Aligned)

---

## 1. Projection Head

### Why is a projection head used?

The base embedding space learned by Word2Vec or Transformer models is optimized for language modeling, not for contrastive alignment.

We introduce a projection head:
\[
x \in \mathbb{R}^d,\quad z = g(x)
\]

This enables:
- preservation of base semantics
- task-specific restructuring of embedding space

---

### How is it trained?

Using NT-Xent Loss:
\[
\mathcal{L}_{NT-Xent} =
-\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}
{\sum_k \exp(\text{sim}(z_i, z_k)/\tau)}
\]

Where:
- \(z_i, z_j\): positive pairs (synonyms or relations)
- \(z_k\): negatives
- \(\tau\): temperature

Full enhanced loss:
\[
\mathcal{L} = \mathcal{L}_{NT-Xent} + \lambda \mathcal{L}_{type}
\]

---

### How is it used?

- Base mode: \(x\)
- Projected mode: \(z = g(x)\)

Projected embeddings are used for:
- retrieval
- link prediction
- relation probing

---

## 2. Ablation Study

### Variants

| Model | Components |
|------|-----------|
| Baseline | None |
| UMLS | Synonym alignment |
| Synonym Only | Clean contrastive |
| Synonym + Type | + classification |
| Synonym + Relation | + relations |
| Full Enhanced | All |

---

### Findings

- Synonyms → semantic clustering
- Types → small classification gain
- Relations → **major improvement in MRR, AUC**

**Trade-off**
- Relations ↑ → retrieval ↑
- Relations ↑ → type F1 ↓

**Key Insight**
> Semantic similarity ≠ relational understanding

---

## 3. Architectures (Repo-Aligned)

---

### 3.1 Word2Vec Base Model

```mermaid
flowchart TD
    A[Corpus] --> B[Context Window]
    B --> C[CBOW / Skip-gram]
    C --> D[Hidden Layer Weights]
    D --> E[Embedding x]
    C --> F[Training Loss]
```

---

### 3.2 Transformer Base Model

```mermaid
flowchart TD
    A[Text Input] --> B[Tokenization]
    B --> C[Embedding Layer]

    C --> D[Transformer Layer 1]
    D --> E[Transformer Layer 2]

    E --> F[Contextual Embedding x]

    F --> G[MLM Head]
    G --> H[Cross Entropy Loss]
```

---

### 3.3 UMLS Alignment Model

```mermaid
flowchart TD
    A[Text Pair] --> B[Encoder]

    B --> C[Embedding x]

    C --> D[Projection Head g(x)]

    D --> E[Projected z]

    E --> F[Similarity]

    F --> G[NT-Xent Loss]

    H[Negatives] --> F
```

---

### 3.4 Enhanced Model (Main Contribution)

```mermaid
flowchart TD
    A[Concept Text] --> B[Encoder]

    B --> C[Embedding x]

    C --> D[Projection Head g(x)]

    D --> E[Projected z]

    %% Contrastive
    E --> F[NT-Xent Loss]
    J[Relation Pairs] --> F

    %% Type branch
    E --> G[Type Classifier]
    G --> H[BCE Loss]

    %% Total
    F --> I[Total Loss]
    H --> I
```

---

### Important Detail

Relations are NOT a separate loss.  
They are injected as additional positive pairs into NT-Xent.

---

### 3.5 Residual Variant (Optional)

```mermaid
flowchart TD
    A[x] --> B[g(x)]
    A --> C[Skip]
    B --> D[Add]
    C --> D
    D --> E[z = x + g(x)]
```

---

## 4. Final Conclusion

- Projection head isolates alignment learning  
- Synonym alignment is insufficient  
- Type supervision gives marginal gains  
- **Relation learning is the dominant factor**

> While synonym learning builds the embedding space, relational supervision gives it meaning.
