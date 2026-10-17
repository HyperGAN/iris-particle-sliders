"""Actual tiny native Iris failure injection; no accelerator allocation or model downloads."""

from copy import deepcopy

import pytest
import torch

from iris_sliders import infer as inference
from iris_sliders import response, train
from iris_sliders.adapters import AdapterBank, new_particles, target_modules
from iris_sliders.card import native_protocol
from iris_sliders.checkpoint import metadata, save_adapter
from iris_sliders.formulation import load_stamp
from iris_sliders.tiny import TinyRuntime

ROWS = [
    {"neutral": "fox", "positive": "fox with golden light"},
    {"neutral": "cabin", "positive": "cabin with golden light"},
]


def configuration():
    return train.configuration(ROWS, steps=2, height=16, width=16, trajectory_steps=4, records_per_pair=1)


def hook_count(runtime):
    return sum(len(module._forward_hooks) for module in runtime.model.modules())


def fail_after_partial_transfer(monkeypatch, runtime):
    original, banks = AdapterBank.to, []

    def transfer(bank, *args, **kwargs):
        assert hook_count(runtime) == runtime.model.cfg.depth
        banks.append(bank)
        original(bank, dtype=torch.float64)
        raise RuntimeError("injected adapter transfer failure")

    monkeypatch.setattr(AdapterBank, "to", transfer)
    return banks


@pytest.mark.parametrize("caller", ["build", "train", "response"])
def test_failed_transfer_closes_hooks_and_allows_same_runtime_retry(monkeypatch, tmp_path, caller):
    runtime = TinyRuntime()
    config = configuration()
    # Give the response caller an actual strict export, prepared before fault injection.
    if caller == "response":
        first = train.build_game(runtime, config)
        packet = {"config": config, "adapter": first.export_state_dict()}
        first.backend.adapter.close()
    base = deepcopy(runtime.model.state_dict())
    rng = torch.get_rng_state().clone()
    assert hook_count(runtime) == 0
    with monkeypatch.context() as fault:
        banks = fail_after_partial_transfer(fault, runtime)
        try:
            with pytest.raises(RuntimeError, match="adapter transfer failure"):
                if caller == "build":
                    train.build_game(runtime, config)
                elif caller == "train":
                    options = {key: value for key, value in config.items() if key != "rows"}
                    train.train(runtime, ROWS, tmp_path, dummy=True, **options)
                else:
                    response.evaluate(runtime, packet)
            assert banks and all(not bank.handles for bank in banks)
            assert hook_count(runtime) == 0
            assert torch.equal(rng, torch.get_rng_state())
            assert all(torch.equal(value, runtime.model.state_dict()[name]) for name, value in base.items())
        finally:
            # Also keep a failing baseline probe from leaving live hooks in other tests.
            for bank in banks:
                bank.close()
    game = train.build_game(runtime, config)
    try:
        assert hook_count(runtime) == runtime.model.cfg.depth
        assert game.step()["step"] == 1
    finally:
        game.backend.adapter.close()
    assert hook_count(runtime) == 0
    assert not torch.cuda.is_initialized()


def test_partial_hook_registration_rolls_back_only_owned_handles_and_allows_retry(monkeypatch):
    runtime = TinyRuntime()
    stamp, _ = load_stamp()
    targets = target_modules(runtime.model)
    observed, registered = [], []
    first, second = targets[0][1], targets[1][1]
    unrelated = first.register_forward_hook(lambda module, inputs, output: observed.append(True))
    existing = set(first._forward_hooks)
    original_register = first.register_forward_hook

    def capture_registration(*args, **kwargs):
        handle = original_register(*args, **kwargs)
        registered.append(handle)
        return handle

    def fail_registration(*args, **kwargs):
        raise RuntimeError("injected native hook registration failure")

    try:
        with monkeypatch.context() as fault:
            fault.setattr(first, "register_forward_hook", capture_registration)
            fault.setattr(second, "register_forward_hook", fail_registration)
            try:
                with pytest.raises(RuntimeError, match="hook registration failure"):
                    AdapterBank(runtime.model, stamp)
                assert registered
                assert hook_count(runtime) == 1
                assert set(first._forward_hooks) == existing
            finally:
                for handle in registered:
                    handle.remove()
        game = train.build_game(runtime, configuration())
        try:
            assert game.step()["step"] == 1
        finally:
            game.backend.adapter.close()
        assert observed and set(first._forward_hooks) == existing and hook_count(runtime) == 1
    finally:
        unrelated.remove()
    assert hook_count(runtime) == 0


@pytest.mark.parametrize("stage", ["transfer", "particles", "metadata"])
def test_mock_inference_setup_closes_hooks_on_every_postattachment_failure(monkeypatch, tmp_path, stage):
    # A tiny native base and real weight-only artifact; only the loader/render surface is mocked.
    runtime = TinyRuntime()
    stamp, declared = load_stamp()
    bank = AdapterBank(runtime.model, stamp)
    path = tmp_path / "adapter.safetensors"
    meta = metadata(
        bank,
        stamp,
        declared,
        runtime.raw["model"],
        base_sha256=runtime.base_sha256,
        validation={"dummy": False},
        protocol=native_protocol(),
    )
    save_adapter(path, bank, new_particles(stamp), meta)
    bank.close()
    monkeypatch.setattr(inference, "IrisRuntime", lambda **kwargs: runtime)
    with monkeypatch.context() as fault:
        banks = []

        def fail(*args, **kwargs):
            assert hook_count(runtime) == runtime.model.cfg.depth
            raise RuntimeError("injected inference setup failure")

        if stage == "transfer":
            banks = fail_after_partial_transfer(fault, runtime)
        elif stage == "particles":
            fault.setattr(inference, "new_particles", fail)
        else:
            fault.setattr(inference, "metadata", fail)
        try:
            with pytest.raises(RuntimeError, match="transfer failure|inference setup failure"):
                inference.main(
                    ["--adapter", str(path), "--prompt", "fox", "--out", str(tmp_path / "image.png")]
                )
            assert hook_count(runtime) == 0
        finally:
            for failed_bank in banks:
                failed_bank.close()
    recovered = AdapterBank(runtime.model, stamp)
    try:
        assert hook_count(runtime) == runtime.model.cfg.depth
    finally:
        recovered.close()
    assert hook_count(runtime) == 0
