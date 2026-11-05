"""Diff V2 + XSA composition and exact standalone regression checks."""
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

import nanochat.flash_attention as fa
import nanochat.gpt as gpt
from nanochat.engine import KVCache
from test_xsa_mudd import make_model


@pytest.fixture(autouse=True)
def cpu_sdpa(monkeypatch):
    monkeypatch.setattr(fa, "USE_FA3", False)
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", torch.float32)


@pytest.fixture
def previous_gpt(monkeypatch):
    # Pin the implementation immediately before Diff V2 + XSA support.
    result = subprocess.run(
        ["git", "show", "abd3c782dbd534055d5a463f2ee1ecdcd6996f1d:./nanochat/gpt.py"],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, check=True)
    module = ModuleType("_pre_diff_v2_xsa_gpt")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(result.stdout, "pre_diff_v2_xsa/gpt.py", "exec"), module.__dict__)
    module.COMPUTE_DTYPE = torch.float32
    return module


@pytest.mark.parametrize("flags", [{}, {"diff_v2": True}, {"xsa": True}])
def test_standalone_initialization_forward_gradients_update_and_cache_unchanged(previous_gpt, flags):
    models = []
    for module in (previous_gpt, gpt):
        torch.manual_seed(42)
        config = module.GPTConfig(sequence_len=16, vocab_size=256, n_layer=2,
                                  n_head=4, n_kv_head=2, n_embd=32, window_pattern="L", **flags)
        with torch.device("meta"):
            model = module.GPT(config)
        model.to_empty(device="cpu")
        model.init_weights()
        models.append(model)
    previous, current = models
    assert previous.state_dict().keys() == current.state_dict().keys()
    for name, value in previous.state_dict().items():
        torch.testing.assert_close(value, current.state_dict()[name], rtol=0, atol=0)
    assert previous.num_scaling_params() == current.num_scaling_params()
    assert previous.estimate_flops() == current.estimate_flops()
    # Activate attention/MLP and XSA so their behavior is observable.
    for model in models:
        torch.manual_seed(7)
        with torch.no_grad():
            for block in model.transformer.h:
                block.attn.c_proj.weight.normal_(std=0.05)
                block.mlp.c_proj.weight.normal_(std=0.05)
                if block.attn.xsa_alpha is not None:
                    block.attn.xsa_alpha.normal_(std=0.5)
    optimizers = [model.setup_optimizer() for model in models]
    assert optimizers[0].state_dict() == optimizers[1].state_dict()
    tokens = torch.arange(16).view(2, 8)
    torch.testing.assert_close(previous(tokens), current(tokens), rtol=0, atol=0)
    for model in models:
        model(tokens, targets=tokens.roll(-1, 1)).backward()
    for old, new in zip(previous.parameters(), current.parameters()):
        torch.testing.assert_close(old.grad, new.grad, rtol=0, atol=0)
    with torch.compiler.set_stance("force_eager"):
        for optimizer in optimizers:
            optimizer.step()
    for old, new in zip(previous.parameters(), current.parameters()):
        torch.testing.assert_close(old, new, rtol=0, atol=0)
    caches = [KVCache(2, 2, 16, 8, 2, "cpu", torch.float32) for _ in models]
    with torch.no_grad():
        for start, end in ((0, 3), (3, 5), (5, 6), (6, 8)):
            outputs = [model(tokens[:, start:end], kv_cache=cache) for model, cache in zip(models, caches)]
            torch.testing.assert_close(*outputs, rtol=0, atol=0)


@pytest.mark.parametrize("heads,kv_heads", [(4, 1), (4, 2), (4, 4), (6, 6)])
def test_combined_formula_and_gradients_match_per_head_reference(monkeypatch, heads, kv_heads):
    model = make_model(active=True, n_layer=2, n_head=heads, n_kv_head=kv_heads,
                       n_embd=heads * 8, diff_v2=True, xsa=True)
    attn = model.transformer.h[1].attn  # includes value embeddings
    captured = {}
    original = fa.flash_attn.flash_attn_func

    def capture(q, k, v, **kwargs):
        captured.update(q=q, k=k, v=v)
        return original(q, k, v, **kwargs)

    monkeypatch.setattr(gpt.flash_attn, "flash_attn_func", capture)
    x = torch.randn(2, 5, heads * 8, requires_grad=True)
    ve = torch.randn(2, 5, kv_heads * 8, requires_grad=True)
    actual = attn(x, ve, (model.cos[:, :5], model.sin[:, :5]), (-1, 0), None, 1)
    lam = attn.lambda_proj(x).sigmoid()
    outputs = []
    for head in range(heads):
        kv_head = head // (heads // kv_heads)
        k, v = captured["k"][:, :, kv_head], captured["v"][:, :, kv_head]
        pair = [F.scaled_dot_product_attention(captured["q"][:, :, 2 * head + partner], k, v,
                                               is_causal=True) for partner in range(2)]
        vn = F.normalize(v, dim=-1, eps=1e-4)
        # Apply the unchanged standalone XSA formula to each retrieval first.
        # Shared K/V and a shared gate make this linear projection commute with Diff V2.
        projected = [y - attn.xsa_alpha[head].tanh() * (y * vn).sum(-1, keepdim=True) * vn
                     for y in pair]
        outputs.append(projected[0] - lam[:, :, head, None] * projected[1])
    expected = attn.c_proj(torch.stack(outputs, dim=2).flatten(2))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    parameters = (x, ve, attn.c_q.weight, attn.c_k.weight, attn.c_v.weight,
                  attn.ve_gate.weight, attn.lambda_proj.weight, attn.xsa_alpha)
    actual_gradients = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.square().sum(), parameters)
    for actual_grad, expected_grad in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(actual_grad).all() and actual_grad.abs().sum() > 0
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-5)


def test_zero_xsa_gate_matches_diff_v2_exactly():
    standalone = make_model(diff_v2=True, n_layer=2, n_kv_head=2)
    combined = make_model(diff_v2=True, xsa=True, n_layer=2, n_kv_head=2)
    for name, value in standalone.state_dict().items():
        torch.testing.assert_close(value, combined.state_dict()[name], rtol=0, atol=0)
    for model in (standalone, combined):
        torch.manual_seed(7)
        with torch.no_grad():
            for block in model.transformer.h:
                block.attn.c_proj.weight.normal_(std=0.05)
                block.mlp.c_proj.weight.normal_(std=0.05)
    tokens = torch.arange(16).view(2, 8)
    torch.testing.assert_close(standalone(tokens), combined(tokens), rtol=0, atol=0)
    for model in (standalone, combined):
        model(tokens, targets=tokens.roll(-1, 1)).backward()
    for name, parameter in standalone.named_parameters():
        torch.testing.assert_close(parameter.grad, combined.get_parameter(name).grad, rtol=0, atol=0)
    for block in combined.transformer.h:
        assert torch.isfinite(block.attn.xsa_alpha.grad).all() and block.attn.xsa_alpha.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_combined_preserves_dtype_and_finite_gradients(monkeypatch, dtype):
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", dtype)
    model = make_model(active=True, n_layer=2, n_kv_head=2, diff_v2=True, xsa=True)
    attn = model.transformer.h[1].attn
    x = torch.randn(2, 8, 32, dtype=dtype, requires_grad=True)
    output = attn(x, None, (model.cos[:, :8], model.sin[:, :8]), (-1, 0), None, 1)
    assert output.shape == x.shape and output.dtype == dtype
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    for parameter in (attn.c_q.weight, attn.lambda_proj.weight, attn.xsa_alpha):
        assert parameter.dtype == torch.float32
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("window", [-1, 2])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("n_loops", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_combined_causality_prefill_and_chunked_decode(monkeypatch, window, kv_heads, n_loops, dtype):
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", dtype)
    model = make_model(active=True, n_layer=2, n_kv_head=kv_heads, n_loops=n_loops,
                       diff_v2=True, xsa=True).eval()
    model.window_sizes = [(window, 0)] * model.eff_depth
    tokens = torch.arange(16).view(2, 8)
    changed = tokens.clone()
    changed[:, 4:] += 32
    cache = KVCache(2, kv_heads, 16, 8, model.eff_depth, "cpu", dtype)
    with torch.no_grad():
        full = model(tokens)
        torch.testing.assert_close(full[:, :4], model(changed)[:, :4], rtol=0, atol=0)
        outputs = [model(tokens[:, start:end], kv_cache=cache)
                   for start, end in ((0, 3), (3, 5), (5, 6), (6, 8))]
    tolerance = dict(rtol=0.03, atol=0.003) if dtype == torch.bfloat16 else dict(rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(torch.cat(outputs, 1), full, **tolerance)
    assert cache.get_pos() == 8 and cache.n_heads == kv_heads


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("backend", ["sdpa", "fa3"])
def test_cuda_combined_backward_and_decode(monkeypatch, backend):
    if backend == "fa3" and not fa.HAS_FA3:
        pytest.skip("FA3 unavailable")
    monkeypatch.setattr(fa, "USE_FA3", backend == "fa3")
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", torch.bfloat16)
    model = make_model(active=True, n_layer=2, n_head=6, n_kv_head=6, n_embd=768,
                       diff_v2=True, xsa=True).to("cuda").eval()
    tokens = torch.arange(8, device="cuda").view(1, 8)
    model(tokens, targets=tokens.roll(-1, 1)).backward()
    for block in model.transformer.h:
        for parameter in (block.attn.lambda_proj.weight, block.attn.xsa_alpha):
            assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    cache = KVCache(1, 6, 16, 128, 2, "cuda", torch.bfloat16)
    with torch.no_grad():
        full = model(tokens)
        decoded = [model(tokens[:, :3], kv_cache=cache)]
        decoded.extend(model(tokens[:, t:t + 1], kv_cache=cache) for t in range(3, 8))
    torch.testing.assert_close(torch.cat(decoded, 1), full, rtol=0.03, atol=0.003)
