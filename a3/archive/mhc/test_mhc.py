"""mHC: doubly stochastic mixing, starting state, gradients, KV-cache agreement, optimizer setup and experiment selection."""
from types import SimpleNamespace

import pytest
import torch

import nanochat.flash_attention as fa
import nanochat.gpt as gpt
from nanochat.engine import KVCache
from nanochat.experiments import select_experiment


@pytest.fixture(autouse=True)
def cpu_sdpa(monkeypatch):
    monkeypatch.setattr(fa, "USE_FA3", False)
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", torch.float32)


def make_model(active=True, **overrides):
    options = dict(sequence_len=32, vocab_size=256, n_layer=6, n_head=4, n_kv_head=4, n_embd=32, window_pattern="L", mhc=True)
    options.update(overrides)
    torch.manual_seed(42)
    with torch.device("meta"):
        model = gpt.GPT(gpt.GPTConfig(**options))
    model.to_empty(device="cpu")
    model.init_weights()
    if active:
        torch.manual_seed(7)
        with torch.no_grad():
            for block in model.transformer.h:
                block.attn.c_proj.weight.normal_(std=0.05)
                block.mlp.c_proj.weight.normal_(std=0.05)
            for layer in model.mhc or []:
                layer.b[:3].fill_(1.0) # gates
    return model


def test_sinkhorn_is_doubly_stochastic_and_cannot_amplify():
    torch.manual_seed(0)
    m = gpt.sinkhorn_log(torch.randn(3, 5, 4, 4)) # 20 iterations converge at this scale, not for extreme logits
    assert (m >= 0).all()
    torch.testing.assert_close(m.sum(-1), torch.ones(3, 5, 4), atol=1e-3, rtol=0)
    torch.testing.assert_close(m.sum(-2), torch.ones(3, 5, 4), atol=1e-5, rtol=0)
    assert torch.linalg.matrix_norm(m, ord=2).max() <= 1 + 1e-3


def test_initial_coefficients():
    model = make_model(active=False)
    idx = torch.randint(0, 256, (2, 8))
    streams = model.transformer.wte(idx).unsqueeze(-2).expand(2, 8, 4, -1)
    for i, layer in enumerate(model.mhc):
        pre, post, res = layer(streams)
        torch.testing.assert_close(pre.sum(-1), torch.ones(2, 8), atol=0.05, rtol=0)
        assert pre.argmax(-1).eq(i % 4).all()
        torch.testing.assert_close(post, torch.ones(2, 8, 4), atol=0.05, rtol=0)
        assert res.diagonal(dim1=-2, dim2=-1).min() > 0.9


def test_all_coefficients_get_gradients_and_residual_scalars_do_not():
    model = make_model()
    idx = torch.randint(0, 256, (2, 16))
    model(idx, torch.randint(0, 256, (2, 16))).backward()
    for layer in model.mhc:
        for name, p in layer.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, name
    assert model.resid_lambdas.grad is None and model.x0_lambdas.grad is None


def test_optimizer_trains_mhc_and_skips_residual_scalars():
    model = make_model()
    optimizer = model.setup_optimizer()
    trained = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert all(g["params"] for g in optimizer.param_groups)
    assert id(model.resid_lambdas) not in trained and id(model.x0_lambdas) not in trained
    assert all(id(p) in trained for p in model.mhc.parameters())


def test_kv_cache_decode_matches_full_forward():
    model = make_model().eval()
    config = model.config
    idx = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        full = model(idx)
        cache = KVCache(batch_size=1, num_heads=config.n_kv_head, seq_len=32, head_dim=config.n_embd // config.n_head,
                        num_layers=config.effective_depth, device="cpu", dtype=torch.float32)
        prefill = model(idx[:, :8], kv_cache=cache)
        last = [model(idx[:, t:t + 1], kv_cache=cache) for t in range(8, 12)]
    torch.testing.assert_close(prefill[:, -1], full[:, 7], atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(last[-1][:, -1], full[:, -1], atol=1e-3, rtol=1e-3)


def test_parameter_accounting_and_flops():
    model, base = make_model(active=False), make_model(active=False, mhc=False)
    counts, base_counts = model.num_scaling_params(), base.num_scaling_params()
    assert counts["transformer_matrices"] == base_counts["transformer_matrices"]
    assert counts["mhc"] == 6 * (3 + 4 + 4 + 16 + 128 * (4 + 4 + 16))
    assert counts["total"] == base_counts["total"] + counts["mhc"]
    assert model.estimate_flops() - base.estimate_flops() == 6 * 6 * 128 * (4 + 4 + 16)


def test_mhc_off_keeps_residual_scalars_trainable():
    base = make_model(active=False, mhc=False)
    assert base.mhc is None and base.resid_lambdas.requires_grad and base.x0_lambdas.requires_grad


@pytest.mark.parametrize("flags", [dict(n_loops=2), dict(mudd=True), dict(mgvr=True), dict(depth_attention=True)])
def test_rejects_unsupported_combinations(flags):
    with pytest.raises(AssertionError):
        gpt.GPT(gpt.GPTConfig(sequence_len=32, vocab_size=256, n_layer=6, n_head=4, n_kv_head=4, n_embd=32, mhc=True, **flags))


def test_experiment_selection():
    args = SimpleNamespace(experiment="mhc", n_loops=1, n_prelude=0, n_coda=0, mlp_activation="relu2", diff_v2=False,
                           depth_attention=False, depth_attention_stride=None, mgvr=False, xsa=False, mudd=False, mhc=False)
    select_experiment(args)
    assert args.mhc
    with pytest.raises(ValueError, match="conflicts"):
        select_experiment(SimpleNamespace(**{**vars(args), "experiment": "baseline"}))
