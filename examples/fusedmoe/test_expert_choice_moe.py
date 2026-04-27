import torch

from example_expert_choice_moe import expert_choice_moe_reference


def _make_inputs(seed=0, num_tokens=8, d_hidden=16, d_ff=32, n_experts=4, device="cpu"):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    x = torch.randn(num_tokens, d_hidden, generator=gen, device=device)
    w_router = torch.randn(d_hidden, n_experts, generator=gen, device=device)
    w_up = torch.randn(n_experts, d_hidden, d_ff, generator=gen, device=device)
    w_gate = torch.randn(n_experts, d_hidden, d_ff, generator=gen, device=device)
    w_down = torch.randn(n_experts, d_ff, d_hidden, generator=gen, device=device)
    return x, w_router, w_up, w_gate, w_down


def test_reference_shape():
    x, w_router, w_up, w_gate, w_down = _make_inputs()
    out, debug = expert_choice_moe_reference(x, w_router, w_up, w_gate, w_down, capacity_factor=2.0, return_debug=True)
    assert out.shape == x.shape
    assert debug["top_indices"].shape[0] == w_up.shape[0]


def test_capacity_per_expert():
    x, w_router, w_up, w_gate, w_down = _make_inputs(num_tokens=10, n_experts=5)
    _, debug = expert_choice_moe_reference(x, w_router, w_up, w_gate, w_down, capacity_factor=1.5, return_debug=True)
    top_indices = debug["top_indices"]
    capacity = debug["capacity"]
    assert top_indices.shape == (w_up.shape[0], capacity)


def test_weight_normalization_for_selected_tokens():
    x, w_router, w_up, w_gate, w_down = _make_inputs(seed=1, num_tokens=12, n_experts=3)
    _, debug = expert_choice_moe_reference(x, w_router, w_up, w_gate, w_down, capacity_factor=1.0, return_debug=True)
    denom = debug["denom"]
    normalized_weight_sums = debug["normalized_weight_sums"]
    selected = denom > 0
    assert torch.allclose(
        normalized_weight_sums[selected],
        torch.ones_like(normalized_weight_sums[selected]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_dropped_token_zero_output():
    device = "cpu"
    num_tokens = 5
    d_hidden = 8
    d_ff = 12
    n_experts = 2

    x = torch.randn(num_tokens, d_hidden, device=device)
    w_router = torch.full((d_hidden, n_experts), -20.0, device=device)
    w_router[0, 0] = 20.0
    w_router[0, 1] = -20.0
    w_up = torch.randn(n_experts, d_hidden, d_ff, device=device)
    w_gate = torch.randn(n_experts, d_hidden, d_ff, device=device)
    w_down = torch.randn(n_experts, d_ff, d_hidden, device=device)

    out, debug = expert_choice_moe_reference(x, w_router, w_up, w_gate, w_down, capacity_factor=0.4, return_debug=True)
    dropped = debug["denom"] == 0
    if torch.any(dropped):
        assert torch.allclose(out[dropped], torch.zeros_like(out[dropped]), atol=1e-6, rtol=1e-6)


def test_tokens_less_than_experts():
    x, w_router, w_up, w_gate, w_down = _make_inputs(seed=2, num_tokens=3, n_experts=10)
    out, debug = expert_choice_moe_reference(x, w_router, w_up, w_gate, w_down, capacity_factor=2.0, return_debug=True)
    assert out.shape == x.shape
    assert debug["capacity"] == 1
    assert debug["top_indices"].shape == (10, 1)


def test_large_capacity_factor():
    x, w_router, w_up, w_gate, w_down = _make_inputs(seed=3, num_tokens=8, n_experts=4)
    out, debug = expert_choice_moe_reference(x, w_router, w_up, w_gate, w_down, capacity_factor=10.0, return_debug=True)
    assert out.shape == x.shape
    assert debug["capacity"] == 8
    assert debug["top_indices"].shape == (4, 8)


def test_float16_precision():
    x_fp32, w_router_fp32, w_up_fp32, w_gate_fp32, w_down_fp32 = _make_inputs(seed=42, num_tokens=50, d_hidden=64, d_ff=128, n_experts=4)
    out_fp32 = expert_choice_moe_reference(x_fp32, w_router_fp32, w_up_fp32, w_gate_fp32, w_down_fp32, capacity_factor=2.0)

    x_fp16 = x_fp32.half()
    w_router_fp16 = w_router_fp32.half()
    w_up_fp16 = w_up_fp32.half()
    w_gate_fp16 = w_gate_fp32.half()
    w_down_fp16 = w_down_fp32.half()
    out_fp16 = expert_choice_moe_reference(x_fp16, w_router_fp16, w_up_fp16, w_gate_fp16, w_down_fp16, capacity_factor=2.0)

    assert out_fp16.shape == out_fp32.shape
    assert not torch.isnan(out_fp16).any()
    assert not torch.isinf(out_fp16).any()
    relative_error = (out_fp16.float() - out_fp32).abs() / (out_fp32.abs().clamp_min(1e-6))
    assert relative_error.mean().item() < 0.1, f"Mean relative error {relative_error.mean().item():.4f} exceeds 10%"


def test_single_token():
    x, w_router, w_up, w_gate, w_down = _make_inputs(seed=5, num_tokens=1, n_experts=4)
    out, debug = expert_choice_moe_reference(x, w_router, w_up, w_gate, w_down, capacity_factor=2.0, return_debug=True)
    assert out.shape == (1, x.shape[1])
    assert debug["capacity"] == 1
    assert debug["denom"].item() > 0


def test_large_scale():
    x, w_router, w_up, w_gate, w_down = _make_inputs(seed=100, num_tokens=256, d_hidden=256, d_ff=512, n_experts=8)
    out, debug = expert_choice_moe_reference(x, w_router, w_up, w_gate, w_down, capacity_factor=2.0, return_debug=True)
    assert out.shape == x.shape
    expected_capacity = min(256, max(1, 256 * 2 // 8))
    assert debug["capacity"] == expected_capacity
