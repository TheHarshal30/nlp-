# Triplet Relation Module

Standalone biomedical relation-learning module built around UMLS triplets and triplet loss.

This module:

- parses `MRREL.RRF` and `MRCONSO.RRF`
- filters UMLS relations to relation-bearing biomedical edges
- loads pretrained `word2vec` or transformer exports
- trains a relation-aware projection space with triplet loss
- exports the learned projection head and metadata
- evaluates disease-to-drug retrieval with Hits@K and MRR

## Layout

```text
triplet_relation_module/
├── data/
│   ├── raw/
│   ├── processed/
│   └── triples/
├── src/
│   ├── data/
│   ├── losses/
│   ├── mining/
│   ├── models/
│   ├── training/
│   └── utils/
├── configs/
│   └── triplet.json
├── scripts/
│   └── train_triplet
├── outputs/
│   ├── logs/
│   └── models/
└── README.md
```

## Supported Base Models

- `word2vec`
- `transformer`

The module consumes the exported model directories produced by `embedding_training_v2`.

## Accepted UMLS Relations

- `may_treat`
- `has_finding_site`
- `has_manifestation`

All other relations are discarded.

## Training

```bash
PYTHONPATH=. python3 triplet_relation_module/scripts/train_triplet \
  --config triplet_relation_module/configs/triplet.json
```

## Outputs

Training writes a model bundle under:

```text
triplet_relation_module/outputs/models/triplet_relation/
├── weights/
│   ├── projection_head.pt
│   ├── relation_embeddings.pt
│   └── base_encoder.pt        # only when fine-tuning
├── metadata.json
└── retrieval_metrics.json
```

## Retrieval Evaluation

The built-in evaluation uses the `may_treat` relation as a disease-to-drug retrieval task.

Metrics:

- `hits@1`
- `hits@5`
- `hits@10`
- `mrr`

The split is done on anchor diseases to avoid leakage between train and test anchors.
