import math

import pytest
import torch

from example_expert_choice_moe import expert_choice_moe, expert_choice_moe_reference


def pytorch_reference_moe(x, w_router, w_up, w_gate, w_down, capacity_factor=2.0, eps=1e-9):
    num_tokens, _ = x.shape
    n_experts = w_up.shape[0]
    capacity = max(1, math.ceil(num_tokens * capacity_factor / n_experts))
    capacity = min(capacity, num_tokens)

    scores = torch.softmax(x.float() @ w_router.float(), dim=-1)
    affinity = scores.transpose(0, 1)
    top_values, top_indices = torch.topk(affinity, k=capacity, dim=-1)

    acc = torch.zeros(num_tokens, x.shape[1], dtype=torch.float32, device=x.device)
    denom = torch.zeros(num_tokens, dtype=torch.float32, device=x.device)

    for e in range(n_experts):
        for c in range(capacity):
            t = int(top_indices[e, c].item())
            h = x[t].float()
            up = h @ w_up[e].float()
            gate = h @ w_gate[e].float()
            ffn = (up * torch.nn.functional.silu(gate)) @ w_down[e].float()
            weight = top_values[e, c]
            acc[t] = acc[t] + weight * ffn
            denom[t] = denom[t] + weight

    out = torch.zeros_like(x)
    selected = denom > 0
    out[selected] = (acc[selected] / denom[selected].unsqueeze(-1).clamp_min(eps)).to(x.dtype)
    return out


def _make_inputs(seed, num_tokens, d_hidden, d_ff, n_experts, device):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    x = torch.randn(num_tokens, d_hidden, generator=gen, device=device)
    w_router = torch.randn(d_hidden, n_experts, generator=gen, device=device)
    w_up = torch.randn(n_experts, d_hidden, d_ff, generator=gen, device=device)
    w_gate = torch.randn(n_experts, d_hidden, d_ff, generator=gen, device=device)
    w_down = torch.randn(n_experts, d_ff, d_hidden, generator=gen, device=device)
    return x, w_router, w_up, w_gate, w_down


def test_regression_reference_matches_independent_impl_case1():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, w_router, w_up, w_gate, w_down = _make_inputs(7, 16, 32, 64, 4, device)

    got = expert_choice_moe_reference(x, w_router, w_up, w_gate, w_down, capacity_factor=2.0)
    expected = pytorch_reference_moe(x, w_router, w_up, w_gate, w_down, capacity_factor=2.0)

    torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)


def test_regression_reference_matches_independent_impl_case2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, w_router, w_up, w_gate, w_down = _make_inputs(11, 9, 24, 48, 3, device)

    got = expert_choice_moe_reference(x, w_router, w_up, w_gate, w_down, capacity_factor=1.3)
    expected = pytorch_reference_moe(x, w_router, w_up, w_gate, w_down, capacity_factor=1.3)

    torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)


def test_hybrid_backend_matches_reference_or_skips():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, w_router, w_up, w_gate, w_down = _make_inputs(19, 8, 16, 32, 2, device)

    reference = expert_choice_moe(x, w_router, w_up, w_gate, w_down, capacity_factor=2.0, backend="reference")
    try:
        hybrid = expert_choice_moe(x, w_router, w_up, w_gate, w_down, capacity_factor=2.0, backend="hybrid")
    except RuntimeError as exc:
        if "hybrid backend is not available yet" in str(exc):
            pytest.skip(str(exc))
        raise

    torch.testing.assert_close(hybrid, reference, rtol=1e-2, atol=1e-2)
