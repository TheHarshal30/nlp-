# Biomedical Embedding Study — Clean GitHub Version

---

## 1. Projection Head

### Why used
Base embeddings are optimized for language modeling, not contrastive alignment.

We use:
x in R^d  
z = g(x)

Projection allows:
- preserving base embedding
- learning task-specific space

---

### Training

NT-Xent Loss:

L = -log( exp(sim(zi, zj)/tau) / sum exp(sim(zi, zk)/tau) )

---

### Usage

- base embedding: x  
- projected embedding: z  

---

## 2. Ablation Study

Models:

- baseline  
- umls  
- synonym only  
- synonym + type  
- synonym + relation  
- full enhanced  

Findings:

- synonyms → clustering  
- types → small gain  
- relations → major improvement  

---

## 3. Architectures

### 3.1 Word2Vec

```mermaid
flowchart TD
A[Corpus] --> B[Context Window]
B --> C[Training Objective]
C --> D[Hidden Layer]
D --> E[Embedding]
```

---

### 3.2 Transformer (Exact from config)

- Layers: 2
- Heads: 4
- Hidden size: 128
- FFN dim: 256
- Max length: 32
- Mask probability: 0.15

```mermaid
flowchart TD
A[Text] --> B[Tokenize]
B --> C[Embedding]
C --> D[Transformer Layer 1]
D --> E[Transformer Layer 2]
E --> F[CLS Representation]
F --> G[MLM Head]
G --> H[Loss]
```

---

### 3.3 Alignment

```mermaid
flowchart TD
A[Input Pair] --> B[Encoder]
B --> C[Embedding]
C --> D[Projection]
D --> E[Projected]
E --> F[Similarity]
F --> G[Contrastive Loss]
```

---

### 3.4 Enhanced Model

```mermaid
flowchart TD
A[Input] --> B[Encoder]
B --> C[Embedding]
C --> D[Projection]
D --> E[Projected]

E --> F[Contrastive Loss]
E --> G[Type Classifier]

G --> H[Type Loss]

F --> I[Total Loss]
H --> I
```

---

### 3.5 Residual Variant

```mermaid
flowchart TD
A[Embedding] --> B[Projection]
A --> C[Skip]
B --> D[Add]
C --> D
D --> E[Output]
```

---

## 4. Conclusion

- projection isolates learning
- synonym alone insufficient
- relations most important
