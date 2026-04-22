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

### Enhanced objective (Word2Vec and Transformer)

Enhanced aligned models (`word2vec_umls_enhanced`, `transformer_umls_enhanced`) combine three signals:

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
- `run_ablation_study`: trains and evaluates controlled ablation variants

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

## 12) Empirical results snapshot

The following results summarize the latest runs shared in this project context.

### 12.1 Standalone downstream evaluation (`embedding_evaluation`)

These metrics come from the standalone evaluator (`entity_linking` + `sts`) on the configured smoke/public splits.

| Model | Entity Linking Acc@1 | Entity Linking Acc@5 | Entity Linking MRR | STS Pearson |
|---|---:|---:|---:|---:|
| `word2vec` | 0.21875 | 0.43750 | 0.32524 | 0.60434 |
| `word2vec_umls` | 0.21875 | 0.43750 | 0.32524 | 0.60434 |
| `transformer` | 0.23438 | 0.46094 | 0.34486 | 0.19218 |
| `transformer_umls` | 0.23438 | 0.43750 | 0.33637 | 0.47289 |

Interpretation:

- `word2vec` and `word2vec_umls` were identical in this run, which indicates no practical downstream effect from the old Word2Vec alignment export in that evaluation mode.
- `transformer_umls` strongly improved STS versus `transformer` (higher semantic similarity correlation).
- Transformer entity-linking differences were modest and mixed, showing that alignment gains are task-dependent.

### 12.2 Relation probing (`scripts/run_relation_probing`)

These metrics evaluate type and relational structure directly.

| Model | Type F1 | P@20 | MRR | AUC |
|---|---:|---:|---:|---:|
| `word2vec` | 0.49311 | 0.02022 | 0.21127 | 0.62464 |
| `word2vec_umls` | 0.49429 | 0.02022 | 0.21127 | 0.62464 |
| `enhanced` (`word2vec_umls_enhanced`) | 0.38857 | 0.03658 | 0.40700 | 0.71501 |

Interpretation:

- `enhanced` substantially improves relation-centric metrics (`P@20`, `MRR`, `AUC`) over both Word2Vec baselines.
- Type macro-F1 dropped for `enhanced`, indicating the current loss balance favors relational structure more than type separability.
- This is an expected tradeoff direction for the present hyperparameter setting (`relation_sampling_ratio`, `type_loss_weight`, warmup schedule).

### 12.3 Projection sanity (Word2Vec UMLS)

After enabling projected inference correctly:

- base shape: `(3, 300)`
- projected shape: `(3, 256)`
- base similarities: related `0.48665`, unrelated `0.32470`
- projected similarities: related `0.21065`, unrelated `0.32129`

Interpretation:

- projection is active and changes geometry (different dimensionality and scores)
- but the old `word2vec_umls` projected space is still not clearly better on simple related/unrelated sanity pairs
- `word2vec_umls_enhanced` is currently the stronger relation-aware variant

## 13) Ablation study protocol

The repository now includes a dedicated ablation runner:

- [run_ablation_study](/home/harshal/nlp%20project%20/TrainWord2Vec/TrainWord2Vec/scripts/run_ablation_study)

### 13.0 Why ablation is required

Baseline-vs-enhanced comparisons tell you **whether** gains exist, but not **which supervision signal caused them**. In this project, enhanced alignment mixes:

- synonym contrastive signal
- semantic type supervision
- relation-pair augmentation

Ablation isolates these components so conclusions are causal, not anecdotal.

### 13.1 Controlled variants and what they test

The runner derives four variants from a single enhanced config:

- `synonym_only`
- `synonym_plus_type`
- `synonym_plus_relation`
- `full_enhanced`

Meaning of each variant:

- `synonym_only`:
  - relation sampling disabled
  - type loss disabled
  - measures pure synonym contrastive alignment
- `synonym_plus_type`:
  - relation sampling disabled
  - type loss enabled
  - isolates contribution of type supervision
- `synonym_plus_relation`:
  - relation sampling enabled
  - type loss disabled
  - isolates contribution of relation augmentation
- `full_enhanced`:
  - relation sampling enabled
  - type loss enabled
  - captures interaction effects between type and relation signals

### 13.2 Experimental controls (important for fair attribution)

Keep these fixed across variants:

- same base checkpoint (`word2vec` or `transformer_fast`)
- same train/validation data artifacts
- same batch size, optimizer, lr, epochs
- same probe settings (`batch_size`, `type_epochs`, limits)
- same seed (or same list of seeds if doing stability runs)

Only the ablation switches should change.

### 13.3 Hypotheses to test

Expected directional behavior:

- if type supervision helps: `synonym_plus_type` > `synonym_only` on Type F1
- if relation augmentation helps: `synonym_plus_relation` > `synonym_only` on P@20 / MRR / AUC
- if signals are complementary: `full_enhanced` best overall
- if signals conflict: one metric family improves while another drops

### 13.4 Metrics and interpretation map

The ablation report focuses on:

- `type_f1`: semantic type separability
- `p@20`: quality of top-k relational retrieval
- `mrr`: ranking quality for positives
- `auc`: binary link discrimination

How to interpret common patterns:

- `type_f1` up, retrieval flat:
  - classifier signal learned, relation geometry not improved
- retrieval up, `type_f1` down:
  - relation constraints dominate type structure
- all up:
  - desirable regime, signals likely synergistic
- all flat:
  - likely inference mismatch, stale export, or weak supervision

### 13.5 Word2Vec ablation (full run)

```bash
PYTHONPATH=. python3 scripts/run_ablation_study \
  --family word2vec \
  --mode all \
  --batch-size 256 \
  --type-epochs 10
```

### 13.6 Transformer ablation (full run)

```bash
PYTHONPATH=. python3 scripts/run_ablation_study \
  --family transformer \
  --mode all \
  --batch-size 256 \
  --type-epochs 10
```

### 13.7 Split execution modes

Train only:

```bash
PYTHONPATH=. python3 scripts/run_ablation_study \
  --family word2vec \
  --mode train
```

Probe only (re-use already trained ablation exports):

```bash
PYTHONPATH=. python3 scripts/run_ablation_study \
  --family word2vec \
  --mode probe \
  --batch-size 256 \
  --type-epochs 10
```

Skip models that already exist:

```bash
PYTHONPATH=. python3 scripts/run_ablation_study \
  --family transformer \
  --mode train \
  --skip-existing
```

### 13.8 Fast smoke ablation

```bash
PYTHONPATH=. python3 scripts/run_ablation_study \
  --family word2vec \
  --mode all \
  --type-epochs 5 \
  --max-type-examples 3000 \
  --max-relation-queries 500 \
  --max-link-pairs 3000
```

### 13.9 Ablation outputs

- Per-family model exports:
  - `embedding_training_v2/outputs/models/ablation_word2vec/`
  - `embedding_training_v2/outputs/models/ablation_transformer/`
- Per-family reports:
  - `results/ablation_word2vec/comparison.json`
  - `results/ablation_word2vec/comparison.md`
  - `results/ablation_transformer/comparison.json`
  - `results/ablation_transformer/comparison.md`

`comparison.json` is machine-friendly; `comparison.md` is report-ready.

### 13.10 Recommended reporting template

For each family, present:

1. one ablation table (`synonym_only`, `synonym_plus_type`, `synonym_plus_relation`, `full_enhanced`)
2. one short paragraph on which signal drove the largest gain
3. one short paragraph on tradeoff (for example type vs relation)
4. optional 3-seed mean/std table for the best two variants

## 14) Entry points (practical)

Train:

```bash
python embedding_training_v2/scripts/train_word2vec --config embedding_training_v2/configs/word2vec.json
python embedding_training_v2/scripts/train_transformer --config embedding_training_v2/configs/transformer.json
python embedding_training_v2/scripts/train_alignment_word2vec --config embedding_training_v2/configs/word2vec_umls.json
python embedding_training_v2/scripts/train_alignment_transformer --config embedding_training_v2/configs/transformer_umls.json
python embedding_training_v2/scripts/train_alignment_word2vec --config embedding_training_v2/configs/word2vec_umls_enhanced.json
python embedding_training_v2/scripts/train_alignment_transformer --config embedding_training_v2/configs/transformer_umls_enhanced.json
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
