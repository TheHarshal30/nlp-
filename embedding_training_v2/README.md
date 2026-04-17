# Embedding Training V2

Clean-room biomedical embedding training module for four MedBench-compatible models:

- `word2vec`
- `word2vec_umls`
- `transformer`
- `transformer_umls`

This package is isolated from the legacy training scripts in the repository.

## Layout

```text
embedding_training_v2/
├── data/
├── configs/
├── src/
├── scripts/
├── outputs/
└── README.md
```

## Model Families

### `word2vec`
- Skip-gram with negative sampling
- Trained on tokenized PubMed abstracts
- Encodes text by mean-pooling token vectors

### `word2vec_umls`
- Loads pretrained `word2vec`
- Runs shared UMLS contrastive alignment with a projection head
- Supports frozen or unfrozen base embeddings

### `transformer`
- Lightweight encoder trained from scratch with masked language modeling
- Encodes text with configurable `cls` or mean pooling

### `transformer_umls`
- Loads pretrained `transformer`
- Uses the same contrastive alignment framework as `word2vec_umls`

## Configs

All scripts take a JSON config file. Example configs are in `configs/`.

## Training

```bash
python embedding_training_v2/scripts/train_word2vec --config embedding_training_v2/configs/word2vec.json
python embedding_training_v2/scripts/train_transformer --config embedding_training_v2/configs/transformer.json
python embedding_training_v2/scripts/train_alignment_word2vec --config embedding_training_v2/configs/word2vec_umls.json
python embedding_training_v2/scripts/train_alignment_transformer --config embedding_training_v2/configs/transformer_umls.json
```

## Export Contract

Each training job writes a MedBench-compatible export under:

```text
embedding_training_v2/outputs/models/<model_name>/
├── weights/
├── model.py
└── metadata.json
```

The exported `model.py` defines a MedBench-style embedder:

```python
class ModelName(BaseEmbedder):
    def load(self, model_path): ...
    def encode(self, texts, batch_size): ...
```

## MedBench Integration

```bash
git clone https://github.com/dz016/medical-entity-linking.git
python scripts/prepare_medbench_models --medbench-root medical-entity-linking
python scripts/run_medbench_benchmarks --medbench-root medical-entity-linking
```

`prepare_medbench_models` copies the trained exports into `medical-entity-linking/models/<model_name>/` using the canonical model names:

- `word2vec`
- `word2vec_umls`
- `transformer`
- `transformer_umls`

It also rewrites `model.py` in each destination folder from the current export contract, so existing trained `*_fast` artifacts can be evaluated without retraining.

`run_medbench_benchmarks` performs:

- model preparation
- preflight encode validation inside the MedBench layout
- `python run_all.py --model <model_name>` for each selected model
- result aggregation into `medical-entity-linking/results/embedding_benchmark_summary.json`
- a markdown comparison table at `medical-entity-linking/results/embedding_benchmark_summary.md`

Example subset run:

```bash
python scripts/run_medbench_benchmarks \
  --medbench-root medical-entity-linking \
  --models transformer transformer_umls
```
