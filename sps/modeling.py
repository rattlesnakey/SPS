from typing import Any
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def render_chat(tokenizer, user_prompt: str, enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def load_model(model_name: str, dtype: str = "bfloat16"):
    dtype_obj = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype_obj,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model


def token_entropy(logits: torch.Tensor) -> float:
    logp = torch.log_softmax(logits.float(), dim=-1)
    p = torch.exp(logp)
    return float(-(p * logp).sum().item())


def boundary_entropy(tokenizer, model, text: str) -> float:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=False)
    return token_entropy(out.logits[0, -1])


def prefix_state(tokenizer, model, text: str, layer_1based: int) -> torch.Tensor:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    with torch.inference_mode():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    return out.hidden_states[layer_1based][0, -1].detach().float().cpu()


def candidate_representation(tokenizer, model, prefix_text: str, candidate: str, layer_1based: int) -> torch.Tensor:
    prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prefix_text + candidate, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    start = prefix_ids.shape[1]
    with torch.inference_mode():
        out = model(input_ids=full_ids, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[layer_1based][0, start:]
    if h.shape[0] == 0:
        h = out.hidden_states[layer_1based][0, -1:]
    return h.mean(dim=0).detach().float().cpu()


def layer_module(model, layer_1based: int):
    return model.model.layers[layer_1based - 1]
