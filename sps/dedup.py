import json
from pathlib import Path
import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer


class CandidateDeduplicator:
    def __init__(self, embedding_model: str, lexical_threshold: float, semantic_threshold: float, batch_size: int):
        self.embedding_model = embedding_model
        self.lexical_threshold = float(lexical_threshold)
        self.semantic_threshold = float(semantic_threshold)
        self.batch_size = int(batch_size)
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        self.encoder = SentenceTransformer(embedding_model, trust_remote_code=True)

    def lexical_similarity(self, a: str, b: str) -> float:
        return float(self.rouge.score(a, b)["rougeL"].fmeasure)

    def lexical_filter(self, candidates: list[str]) -> list[str]:
        kept = []
        for text in candidates:
            text = str(text).strip()
            if not text:
                continue
            if any(self.lexical_similarity(text, old) > self.lexical_threshold for old in kept):
                continue
            kept.append(text)
        return kept

    def semantic_filter(self, candidates: list[str]) -> list[str]:
        if len(candidates) <= 1:
            return candidates
        embeddings = self.encoder.encode(
            candidates,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        kept_indices = []
        for i in range(len(candidates)):
            if kept_indices:
                similarities = embeddings[kept_indices] @ embeddings[i]
                if float(np.max(similarities)) > self.semantic_threshold:
                    continue
            kept_indices.append(i)
        return [candidates[i] for i in kept_indices]

    def deduplicate(self, candidates: list[str]) -> list[str]:
        return self.semantic_filter(self.lexical_filter(candidates))


def load_thresholds(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return {
        "lexical_threshold": float(obj["lexical_threshold"]),
        "semantic_threshold": float(obj["semantic_threshold"]),
    }
