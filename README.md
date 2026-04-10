# CARE
## Overview
This repository provides code to train and run CARE designed for Chinese medical retrieval tasks. The system trains a compact query encoder and a larger document encoder.

## Requirements
- Python 3.8+
- packages: `torch`, `transformers`, `huggingface_hub`, `numpy`, `tqdm`, `FlagEmbedding`.
```
pip install -r requirements.txt
```


## Training
1. `scripts/stage1-query-alignment.sh` are training scripts for stage I. Set `ROOT` and model paths (query encoder and document encoder seperately). `FIX_DOC_ENCODER` should set to be `True` at this stage.
2. `scripts/stage2-joint-finetuning.sh` are scripts for stage II. Model checkpoints of query encoder are from stage I, while model path of document encoder are same as stage I. 

Both scripts requires set `ROOT`, `MODEL_QUERY`, `MODEL_DOC`, `TRAIN_DATA`, `SAVE_DIR`, and cluster env variables (`WORLD_SIZE`, `RANK`, `MASTER_ADDR`). The training entrypoint is `train/main.py`

```bash
export ROOT=/path/to/project
cd $ROOT/Medical-Asymmetric-Retriever/train

bash ../scripts/stage1-query-alignment.sh

bash ../scripts/stage2-joint-finetuning.sh
```

## Inference
Use `inference/asymmetric.py::CARE` to load encoders and compute embeddings. Example:
```python
from inference.asymmetric import CARE
import numpy as np
model_name_or_path_query = "path/to/query/encoder"
model_name_or_path_doc = "path/to/document/encoder"
care = CARE(
    model_name_or_path_query=model_name_or_path_query,
    model_name_or_path_doc=model_name_or_path_doc,
    trust_remote_code=True,
    use_fp16=False,
    normalize_embeddings=True,
    query_batch_size=2,
    passage_batch_size=2,
)
queries = [
    "什么是高血压？"
]
corpus = [
    "高血压是指动脉血压持续升高，通常指收缩压≥140mmHg和/或舒张压≥90mmHg。"
]
query_embeddings = care.encode_queries(queries, task_name='retrieval')
print("Query Embeddings:", query_embeddings, query_embeddings.shape)
corpus_embeddings = care.encode_corpus(corpus, task_name='retrieval')
print("Corpus Embeddings:", corpus_embeddings, corpus_embeddings.shape)

scores = np.dot(query_embeddings, corpus_embeddings.T)
print("Similarity Scores:", scores)
```

## Per file descriptions
- `inference/asymmetric.py`:
  - `CARE` inference wrapper that loads separate query and document tokenizers and encoders.

- `scripts/stage1-query-alignment.sh`:
  - Example shell script for stage-1 query alignment training.
- `scripts/stage2-joint-finetuning.sh`:
  - Example shell script for stage-2 joint finetuning.

- `train/arguments.py`:
  - Defines dataclasses for model, data and training arguments.
- `train/dataset.py`:
  - Dataset classes and collators used during training.
- `train/load_model.py`:
  - Utilities to construct/load the document encoder.
- `train/main.py`:
  - Training entrypoint. Parses arguments into dataclasses, initializes `AsymmetricEmbedderRunner`, and starts training.
- `train/modeling.py`:
  - Contains `AsymmetricEmbedderModel` which wraps the query and document encoders.
- `train/runner.py`:
  - Runner that wires together tokenizers, base models, the `AsymmetricEmbedderModel`, and data collators.
- `train/trainer.py`:
  - Custom trainer.

- `data`:
  - Sampled training data examples for stage-1 and stage-2.
  - `stage1`: Sampled data for stage-1. `stage1-query-align-q.jsonl` are query-side triples, and `stage1-query-align-doc.jsonl` are document-side triples.
  - `stage2`: Sampled data for stage-2. `stage2-medteb-retrieval.jsonl` are sampled from MedTEB retrieval train part. `stage2-medteb-sts.jsonl` are sampled from MedTEB STS train part.
  
## Reproducibility
Use `scripts/` as templates to reproduce experimental settings. Ensure paths and cluster environment variables are set correctly.

## License
License: CC-BY-NC-SA-4.0.


