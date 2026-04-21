# Biomedical Embedding System: Theory and Architecture

This repository implements a biomedical representation learning system that moves from raw biomedical text to task-evaluable embedding models. The goal is not only to produce embeddings that are distributionally good, but to inject biomedical structure (synonymy, semantic types, and relations) so downstream retrieval and reasoning tasks improve in measurable ways.

This README explains the system conceptually first, then maps theory to modules and runnable workflows.

## 1) Problem framing

General-purpose embeddings often encode lexical co-occurrence but miss biomedical ontology structure. In biomedical NLP, that gap matters because:

- many concepts are multi-surface (`myocardial infarction`, `heart attack`)
- concepts are typed (disease, drug, anatomy, symptom)
- concepts are linked by meaningful relations (disease `has_manifestation` symptom, drug `may_treat` disease)

The system therefore combines:

1. distributional pretraining from PubMed text
2. ontology-supervised alignment from UMLS
3. explicit evaluation under identical inference contracts

## 2) Representation learning objective

The project supports four base model families:

- `word2vec`
- `word2vec_umls`
- `transformer`
- `transformer_umls`

and an enhanced branch:

- `word2vec_umls_enhanced` (type + relation aware)

### Core principle

Baseline models learn semantic proximity from corpus context. Aligned models add supervised constraints so biomedical equivalents and related concepts move closer in embedding space.

### Contrastive alignment

For aligned models, training uses NT-Xent contrastive loss over positive pairs:

- anchor and positive should have high similarity
- all other in-batch samples act as negatives

This creates a geometry where UMLS-confirmed concept pairs are more consistently close.

### Enhanced objective (Word2Vec only)

`word2vec_umls_enhanced` combines three signals:

- synonym contrastive loss (retain lexical equivalence)
- relation-pair contrastive augmentation (inject cross-concept structure)
- type classification loss (multi-label semantic type prediction)

Conceptually:

- `L = L_ntxent + lambda * L_type`
- `lambda` supports warmup so the classifier does not dominate early training

## 3) Why projection heads exist

The alignment stage learns a projection function `g(x)` that maps base embeddings into alignment space.

This repository supports two inference modes:

- `base`: use original embedding space
- `projected`: use `g(x)` space

This distinction is crucial because many alignment failures in practice come from training with projection and evaluating without projection. The codebase now exposes this explicitly in metadata and evaluation toggles so alignment effects are observable rather than silently discarded.

## 4) Data semantics and UMLS roles

UMLS resources provide complementary supervision:

- `MRCONSO.RRF`: concept names / lexical surfaces
- `MRSTY.RRF`: semantic types (multi-label)
- `MRREL.RRF`: typed relations

### How they are used

- Synonym alignment pairs are extracted from shared-CUI aliases (via `MRCONSO`)
- Type supervision builds `CUI -> [types]` multi-label targets
- Relation supervision builds `(anchor, relation, positive)` pairs for selected meaningful biomedical relations

The enhanced pipeline filters noisy relation families and retains clinically meaningful edges (for example `may_treat`, `has_manifestation`, `finding_site_of`).

## 5) Module-by-module architecture

### `embedding_training_v2/`

This is the clean training/export core.

Theoretical role:

- learn baseline distributional embeddings
- apply ontology-aware alignment constraints
- export self-contained inference artifacts

Important internal components:

- `src/backbones.py`: base encoders (Word2Vec, transformer)
- `src/alignment.py`: contrastive and enhanced alignment logic
- `src/umls_enhanced.py`: type/relation dataset construction from UMLS
- `src/export.py`: reproducible model contract export
- `src/tasks.py`: orchestrates training tasks and artifact checks

### `scripts/`

Operational reliability layer.

- `validate_embeddings`: structural + numerical sanity checks before benchmarking
- `debug_alignment_effect`: verifies base vs projected behavior is numerically different when expected
- MedBench integration helpers for external benchmark flow
- `run_relation_probing`: runs type/relation probes and writes comparison tables

### `embedding_evaluation/`

Standalone local benchmark layer with no dependency on external benchmark repositories.

Theoretical role:

- hold evaluation constant while swapping models/inference modes
- measure retrieval/classification behavior under identical preprocessing and scoring

Tasks:

- entity linking (Acc@1, Acc@5, MRR)
- STS (Pearson)
- simplified NLI (accuracy, macro F1)
- relation probing (type F1, P@20, MRR, ROC-AUC)

### `triplet_relation_module/`

Separate experimental path for relation-centric embedding shaping with triplet learning. This is intentionally decoupled from the main synonym-alignment path.

## 6) Export contract and reproducibility

All trained models are exported as loadable directories:

```text
<model_dir>/
├── weights/
├── model.py
└── metadata.json
```

`model.py` must expose:

```python
class ModelName(BaseEmbedder):
    def load(self, model_path): ...
    def encode(self, texts, batch_size): ...
```

Why this matters:

- decouples training code from evaluation code
- lets benchmarking tools consume models uniformly
- makes inference behavior explicit through metadata (`inference_mode`, projection flags, dimensions, etc.)

## 7) Evaluation philosophy

The project uses layered validation to reduce silent failures.

### Layer A: numerical sanity

`validate_embeddings` checks:

- loadability
- shape/dtype correctness
- NaN/Inf absence
- non-zero vectors
- non-collapsed variance
- similarity sanity
- batch determinism

### Layer B: behavior sanity

`debug_alignment_effect` checks whether alignment materially changes embedding behavior under projected inference.

### Layer C: downstream utility

Task metrics (entity linking, STS, NLI, relation probes) determine whether geometric changes are useful, not just numerically different.

## 8) Key design choices and tradeoffs

### Freeze vs fine-tune base encoder

- `freeze_base=true`: stable base semantics, cheaper training, safer updates
- `freeze_base=false`: higher capacity but higher collapse/drift risk

### Projection at inference

- `base` mode gives baseline comparability
- `projected` mode reflects what alignment optimized
- both are needed for honest attribution of gains

### Relation augmentation strength

Too much relation sampling can drown synonym signal; too little gives negligible relational structure gains. The config controls this balance explicitly.

### Type loss warmup

Gradual introduction of type supervision prevents early overfitting to frequent classes and improves training stability.

## 9) Typical end-to-end workflow

1. Train baseline models.
2. Run UMLS alignment (`word2vec_umls`, `transformer_umls`, optionally enhanced Word2Vec).
3. Validate exported artifacts (`scripts/validate_embeddings`).
4. Probe alignment effect (`scripts/debug_alignment_effect`).
5. Run standalone evaluation and/or MedBench integration.
6. Compare base vs projected and baseline vs aligned results.

## 10) Critical file dependencies

For UMLS-driven stages:

- `META/MRCONSO.RRF` required for synonym extraction
- `META/MRSTY.RRF` required for type-supervised enhanced alignment
- `META/MRREL.RRF` required for relation-aware enhanced alignment and triplet workflows

Missing files should be treated as hard blockers, not soft warnings.

## 11) What “success” means in this system

A successful run is not just “training completed.” It means:

- exports are structurally valid and numerically healthy
- aligned models differ from baseline in expected directions
- projection/base toggles behave consistently
- downstream metrics are reproducible and comparable under fixed evaluation conditions
- enhanced alignment improves at least one of:
  - type prediction quality
  - relation retrieval quality
  - entity-linking sensitivity

## 12) Entry points (practical)

Train:

```bash
python embedding_training_v2/scripts/train_word2vec --config embedding_training_v2/configs/word2vec.json
python embedding_training_v2/scripts/train_transformer --config embedding_training_v2/configs/transformer.json
python embedding_training_v2/scripts/train_alignment_word2vec --config embedding_training_v2/configs/word2vec_umls.json
python embedding_training_v2/scripts/train_alignment_transformer --config embedding_training_v2/configs/transformer_umls.json
python embedding_training_v2/scripts/train_alignment_word2vec --config embedding_training_v2/configs/word2vec_umls_enhanced.json
```

Validate:

```bash
python scripts/validate_embeddings
PYTHONPATH=. python3 scripts/debug_alignment_effect --model-path embedding_training_v2/outputs/models/word2vec_umls_enhanced
```

Probe:

```bash
PYTHONPATH=. python3 scripts/run_relation_probing \
  --model word2vec=embedding_training_v2/outputs/models/word2vec \
  --model word2vec_umls=embedding_training_v2/outputs/models/word2vec_umls \
  --model enhanced=embedding_training_v2/outputs/models/word2vec_umls_enhanced \
  --cui-to-type embedding_training_v2/data/cui_to_type.json \
  --relation-pairs embedding_training_v2/data/relation_pairs.json
```

Standalone evaluation:

```bash
PYTHONPATH=. python3 embedding_evaluation/scripts/prepare_public_datasets
PYTHONPATH=. python3 embedding_evaluation/scripts/run_all --config embedding_evaluation/configs/public_eval_transformer_umls.json
```
