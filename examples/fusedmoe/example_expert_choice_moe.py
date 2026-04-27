import math
from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F


def _compute_capacity(num_tokens: int, n_experts: int, capacity_factor: float) -> int:
    raw_capacity = math.ceil(num_tokens * capacity_factor / n_experts)
    return max(1, min(num_tokens, raw_capacity))


def _validate_shapes(
    x: torch.Tensor,
    w_router: torch.Tensor,
    w_up: torch.Tensor,
    w_gate: torch.Tensor,
    w_down: torch.Tensor,
) -> Tuple[int, int, int, int]:
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (N, H), got shape {tuple(x.shape)}")

    num_tokens, d_hidden = x.shape
    if w_router.shape[0] != d_hidden:
        raise ValueError(f"w_router first dim must match d_hidden ({d_hidden}), got {tuple(w_router.shape)}")

    n_experts = w_router.shape[1]
    if w_up.shape[:2] != (n_experts, d_hidden):
        raise ValueError(f"w_up must have shape (E, H, F)=({n_experts}, {d_hidden}, F), got {tuple(w_up.shape)}")
    if w_gate.shape[:2] != (n_experts, d_hidden):
        raise ValueError(f"w_gate must have shape (E, H, F)=({n_experts}, {d_hidden}, F), got {tuple(w_gate.shape)}")

    d_ff = w_up.shape[2]
    if w_gate.shape[2] != d_ff:
        raise ValueError(f"w_gate d_ff must match w_up d_ff ({d_ff}), got {w_gate.shape[2]}")
    if w_down.shape != (n_experts, d_ff, d_hidden):
        raise ValueError(f"w_down must have shape (E, F, H)=({n_experts}, {d_ff}, {d_hidden}), got {tuple(w_down.shape)}")

    return num_tokens, d_hidden, n_experts, d_ff


def expert_choice_moe_reference(
    x: torch.Tensor,
    w_router: torch.Tensor,
    w_up: torch.Tensor,
    w_gate: torch.Tensor,
    w_down: torch.Tensor,
    capacity_factor: float = 2.0,
    eps: float = 1e-9,
    return_debug: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """Reference Expert-Choice MoE implementation in PyTorch.

    Routing uses w_router while FFN uses (w_up, w_gate, w_down).
    Dropped tokens (denom == 0) output zeros.

    Note: Accumulation is performed in float32 for numerical stability,
    then cast back to input dtype for output.
    """
    num_tokens, _, n_experts, _ = _validate_shapes(x, w_router, w_up, w_gate, w_down)
    capacity = _compute_capacity(num_tokens, n_experts, capacity_factor)

    router_scores = torch.softmax(x.float() @ w_router.float(), dim=-1)  # (N, E)
    affinity = router_scores.transpose(0, 1)  # (E, N)
    top_values, top_indices = torch.topk(affinity, k=capacity, dim=-1)

    acc = torch.zeros(num_tokens, x.shape[1], dtype=torch.float32, device=x.device)
    denom = torch.zeros(num_tokens, dtype=torch.float32, device=x.device)
    normalized_weight_sums = torch.zeros(num_tokens, dtype=torch.float32, device=x.device)

    for e in range(n_experts):
        token_indices = top_indices[e]  # (capacity,)
        expert_input = x[token_indices].float()  # (capacity, H)

        up = expert_input @ w_up[e].float()
        gate = expert_input @ w_gate[e].float()
        ffn = (up * F.silu(gate)) @ w_down[e].float()  # (capacity, H)

        weights = top_values[e]  # (capacity,)
        weighted_ffn = ffn * weights.unsqueeze(-1)

        acc.index_add_(0, token_indices, weighted_ffn)
        denom.index_add_(0, token_indices, weights)

    out = torch.zeros_like(x)
    selected = denom > 0
    if selected.any():
        out[selected] = (acc[selected] / denom[selected].unsqueeze(-1).clamp_min(eps)).to(x.dtype)

    if return_debug:
        for e in range(n_experts):
            token_indices = top_indices[e]
            weights = top_values[e]
            norm_weights = weights / denom[token_indices].clamp_min(eps)
            normalized_weight_sums.index_add_(0, token_indices, norm_weights)

    if not return_debug:
        return out

    debug = {
        "capacity": capacity,
        "top_indices": top_indices,
        "top_values": top_values,
        "denom": denom.to(x.dtype),
        "normalized_weight_sums": normalized_weight_sums.to(x.dtype),
    }
    return out, debug


def expert_choice_moe(
    x: torch.Tensor,
    w_router: torch.Tensor,
    w_up: torch.Tensor,
    w_gate: torch.Tensor,
    w_down: torch.Tensor,
    capacity_factor: float = 2.0,
    eps: float = 1e-9,
    backend: str = "reference",
):
    if backend == "reference":
        return expert_choice_moe_reference(
            x,
            w_router,
            w_up,
            w_gate,
            w_down,
            capacity_factor=capacity_factor,
            eps=eps,
            return_debug=False,
        )

    if backend == "hybrid":
        raise RuntimeError("hybrid backend is not available yet")

    raise ValueError(f"unknown backend: {backend}")
