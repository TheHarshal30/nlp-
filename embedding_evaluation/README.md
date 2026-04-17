# Embedding Evaluation

Minimal standalone benchmark runner for trained biomedical embedding models.

Tasks:

- Entity Linking
- Semantic Similarity (STS)
- Natural Language Inference (NLI)

This module does not depend on external evaluation repositories. It consumes local model exports and local dataset files.

## Layout

```text
embedding_evaluation/
├── data/
├── src/
│   ├── entity_linking/
│   ├── models/
│   ├── nli/
│   ├── runner/
│   ├── sts/
│   └── utils/
├── configs/
├── scripts/
│   └── run_all
├── results/
└── README.md
```

## Model Interface

All models must provide:

```python
class BaseEmbedder:
    def load(self, model_path): ...
    def encode(self, texts, batch_size): ...
```

The evaluator loads exported `model.py` files from local model directories.

## Dataset Formats

The runner expects user-provided local files and supports lightweight TSV or JSONL inputs.

### Entity Linking

Knowledge base:

- TSV: `entity_id<TAB>canonical_name`
- JSONL: `{"entity_id": "...", "name": "..."}`

Queries:

- TSV: `mention<TAB>entity_id`
- JSONL: `{"mention": "...", "entity_id": "..."}`

### STS

- TSV: `sentence1<TAB>sentence2<TAB>score`
- JSONL: `{"sentence1": "...", "sentence2": "...", "score": 3.5}`

### NLI

- TSV: `premise<TAB>hypothesis<TAB>label`
- JSONL: `{"premise": "...", "hypothesis": "...", "label": "entailment"}`

## Run

```bash
PYTHONPATH=. python3 embedding_evaluation/scripts/run_all \
  --config embedding_evaluation/configs/example.json
```

## Public Dataset Preparation

For publicly mirrored datasets, you can download and normalize local TSV files with:

```bash
pip install datasets
PYTHONPATH=. python3 embedding_evaluation/scripts/prepare_public_datasets
```

This writes normalized files under:

```text
embedding_evaluation/data/public/
├── entity_linking/
├── nli/
├── sts/
└── download_summary.json
```

Current public sources used by the helper:

- `ncbi/ncbi_disease`
- `bigbio/bc5cdr`
- `bigbio/biosses`
- `tasksource/nli4ct`

The downloader uses lightweight schema heuristics. If a source dataset changes schema, the script will need to be adjusted.

## Results

Results are saved to:

```text
embedding_evaluation/results/<model_name>.json
```
