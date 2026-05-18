# This file centralizes paths, dataset size, and model definitions

from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder

CORPUS_PATH = Path("src/global_corpus.pkl")
SAMPLE_PATH = Path("src/dataset.csv")

DATA_SIZE = 1000
TRAIN_RATIO = 0.8

EMBEDDER = SentenceTransformer("BAAI/bge-small-en-v1.5")
CROSS_ENCODER = CrossEncoder("BAAI/bge-reranker-base")