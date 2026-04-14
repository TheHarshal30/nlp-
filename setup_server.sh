#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m nltk.downloader punkt

cat <<'EOF'
Environment ready.

Activate it with:
source .venv/bin/activate

Pipeline commands:
python training/01_train_word2vec.py --abstracts data/pubmed --output models/word2vec/weights/word2vec.bin
python training/02_extract_umls_pairs.py --mrconso META/MRCONSO.RRF --vocab_bin models/word2vec/weights/word2vec.bin --pairs_out data/umls_pairs.txt --vocab_out models/word2vec_umls/weights/umls_vocab.json
python training/03_align_ntxent.py --w2v_bin models/word2vec/weights/word2vec.bin --pairs data/umls_pairs.txt --output models/word2vec_umls/weights/word2vec_umls.bin
EOF
