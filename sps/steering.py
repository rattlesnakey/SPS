import numpy as np
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from .modeling import layer_module


class DelimiterStop(StoppingCriteria):
    def __init__(self, tokenizer, start_length: int, delimiter: str):
        self.tokenizer = tokenizer
        self.start_length = int(start_length)
        self.delimiter = delimiter

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0, self.start_length:], skip_special_tokens=False)
        return self.delimiter in text


class LayerSteering:
    def __init__(self, model, layer_1based: int):
        self.model = model
        self.layer_1based = int(layer_1based)
        self.direction = None
        self.alpha = 0.0
        self.handle = layer_module(model, self.layer_1based).register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self.direction is None or self.alpha == 0.0:
            return output
        direction = self.direction.to(output[0].device, dtype=output[0].dtype)
        h = output[0].clone()
        h[:, -1, :] = h[:, -1, :] + self.alpha * direction
        return (h,) + tuple(output[1:])

    def set(self, direction: np.ndarray, alpha: float):
        self.direction = torch.tensor(direction, dtype=torch.float32)
        self.alpha = float(alpha)

    def clear(self):
        self.direction = None
        self.alpha = 0.0

    def close(self):
        self.handle.remove()


def entropy_percentile(entropy_reference: np.ndarray, value: float) -> float:
    ref = np.asarray(entropy_reference, dtype=np.float64)
    return float(np.mean(ref <= float(value)))


def entropy_adaptive_strength(entropy_reference: np.ndarray, value: float, threshold: float, alpha_min: float) -> tuple[float, float, float]:
    percentile = entropy_percentile(entropy_reference, value)
    threshold_percentile = entropy_percentile(entropy_reference, threshold)
    denominator = max(1.0 - threshold_percentile, 1e-12)
    normalized = float(np.clip((percentile - threshold_percentile) / denominator, 0.0, 1.0))
    alpha = float(alpha_min + (1.0 - alpha_min) * normalized)
    return alpha, percentile, threshold_percentile


def sample_direction(bank: dict, state: np.ndarray, rng: np.random.Generator):
    centroids = bank["region_centroids"]
    region = int(np.argmin(np.linalg.norm(centroids - state[None, :], axis=1)))
    vectors = bank["vectors_by_region"][region]
    if len(vectors) == 0:
        raise RuntimeError(f"No steering vectors in region {region}")
    index = int(rng.integers(0, len(vectors)))
    return region, index, vectors[index]


def generate_step(tokenizer, model, input_ids: torch.Tensor, controller: LayerSteering, generation_cfg: dict, delimiter: str, max_step_tokens: int):
    start = input_ids.shape[1]
    stopping = StoppingCriteriaList([DelimiterStop(tokenizer, start, delimiter)])
    output = model.generate(
        input_ids=input_ids,
        do_sample=True,
        temperature=float(generation_cfg["temperature"]),
        top_p=float(generation_cfg["top_p"]),
        top_k=int(generation_cfg["top_k"]),
        max_new_tokens=int(max_step_tokens),
        stopping_criteria=stopping,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = output[0, start:]
    text = tokenizer.decode(new_ids, skip_special_tokens=False)
    return output, text
