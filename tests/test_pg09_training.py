"""Released shared GAN, actual tiny native Iris, synthetic conditioning, CPU only."""

import json
from copy import deepcopy
from importlib.metadata import version

import pytest
import torch
from particle_sliders import FormulationGame
from particle_sliders.pg09 import ConditionalPG09

from iris_sliders import response
from iris_sliders import train as training
from iris_sliders.teacher import load_prompts
from iris_sliders.tiny import TinyRuntime


def options(**changes):
    return {
        "steps": 5,
        "batch": 2,
        "height": 16,
        "width": 16,
        "trajectory_steps": 4,
        "records_per_pair": 2,
        **changes,
    }


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_tree(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(equal_tree(a, b) for a, b in zip(left, right))
        )
    return left == right


def assert_no_hooks(runtime):
    assert all(not module._forward_hooks for module in runtime.model.modules())


@pytest.mark.parametrize("preset", ["bispf", "bispf_optimized"])
@pytest.mark.parametrize("critic", ["field", "hndl"])
def test_actual_shared_gan_owns_routed_bank_updates_and_balances_native_rows(monkeypatch, preset, critic):
    if critic == "hndl":
        pytest.importorskip("hndl")

    def forbidden(*args, **kwargs):
        raise AssertionError("the shared PG response trainer must not call legacy or MSE training")

    monkeypatch.setattr(FormulationGame, "__init__", forbidden)
    monkeypatch.setattr(torch.nn.functional, "mse_loss", forbidden)
    monkeypatch.setattr(torch.nn.MSELoss, "forward", forbidden)
    runtime = TinyRuntime()
    base = {name: value.detach().clone() for name, value in runtime.model.named_parameters()}
    game = training.build_game(
        runtime, training.configuration(load_prompts(), **options(preset=preset, critic=critic))
    )
    try:
        assert type(game) is ConditionalPG09
        assert version("particlegan") == game._signature["pg_version"] == "0.9.1"
        assert game.response_shape == (1, 3, 16, 16)
        assert game._signature["resolved_penalty_domain"] == "image"
        assert game._signature["objective"] == "g_adv+public_prior;d_adv+public_penalty"
        assert game._signature["generator_arm"] == "routed_particle_gan"
        assert game._signature["backend"]["cfg_adapter_branch"] == "conditional_only"
        assert game.prior.z is game.backend.particles is game.network.particles
        owned = [p for group in game.opt_g.param_groups for p in group["params"]]
        assert len(owned) == len({id(p) for p in owned})
        assert {id(p) for p in owned} == {id(p) for p in game.network.parameters()}
        assert sum(p is game.prior.z for p in owned) == 1
        assert not {id(p) for p in owned}.intersection(id(p) for p in runtime.model.parameters())
        initial_up = [layer.up.weight.detach().clone() for layer in game.backend.adapter.layers]
        initial_particles = game.prior.z.detach().clone()
        for index in range(1, 6):
            stats = game.step()
            assert stats["step"] == index
            assert stats["g_loss"] == pytest.approx(stats["g_adv"] + stats["prior_regularization"], abs=1e-7)
            assert stats["d_loss"] == pytest.approx(stats["d_adv"] + stats["d_pen"], abs=1e-7)
            assert stats["prior_regularization"] > 0
            for phase in ("g", "d"):
                counts = game.exposures[phase]
                assert int(counts.sum()) == index * game.batch_size
                assert int(counts.max() - counts.min()) <= 1
                assert len(stats[phase + "_rows"]) == game.batch_size
        assert all(
            torch.equal(parameter, base[name]) and not parameter.requires_grad and parameter.grad is None
            for name, parameter in runtime.model.named_parameters()
        )
        assert any(
            not torch.equal(layer.up.weight, old)
            for layer, old in zip(game.backend.adapter.layers, initial_up)
        )
        assert not torch.equal(game.prior.z, initial_particles)
        assert game.prior.z.grad is not None and torch.isfinite(game.prior.z.grad).all()
        response_gradient = torch.autograd.grad(
            game.backend.generated(game.rows[0]).square().sum(), game.prior.z
        )[0]
        assert torch.isfinite(response_gradient).all() and torch.count_nonzero(response_gradient)
        with pytest.raises(RuntimeError, match="budget exhausted"):
            game.step()
    finally:
        game.backend.adapter.close()
    assert_no_hooks(runtime)


@pytest.mark.parametrize("preset", ["bispf", "bispf_optimized"])
@pytest.mark.parametrize("critic", ["field", "hndl"])
def test_five_updates_equal_two_plus_three_resume_through_fourth_penalty(tmp_path, preset, critic):
    if critic == "hndl":
        pytest.importorskip("hndl")
    rows = load_prompts()
    settings = options(preset=preset, critic=critic)
    training.train(TinyRuntime(), rows, tmp_path / "full", dummy=True, **settings)
    training.train(TinyRuntime(), rows, tmp_path / "split", dummy=True, stop_after=2, **settings)
    training.train(
        TinyRuntime(), rows, tmp_path / "split", dummy=True, resume=tmp_path / "split/training.pt", **settings
    )
    full = torch.load(tmp_path / "full/training.pt", weights_only=True)
    split = torch.load(tmp_path / "split/training.pt", weights_only=True)
    assert equal_tree(full, split)
    assert full["game"]["completed_steps"] == 5
    assert full["game"]["opt_d"]["regularizer"]["record"]["calls"] == 5
    assert equal_tree(
        torch.load(tmp_path / "full/response-adapter.pt", weights_only=True),
        torch.load(tmp_path / "split/response-adapter.pt", weights_only=True),
    )
    full_log = [json.loads(line) for line in (tmp_path / "full/train.jsonl").read_text().splitlines()]
    split_log = [json.loads(line) for line in (tmp_path / "split/train.jsonl").read_text().splitlines()]
    assert full_log == split_log
    assert full_log[3]["step"] == 4 and full_log[3]["d_pen"] > 0
    assert json.loads((tmp_path / "full/run.json").read_text())["validation"]["quality_validated"] is False


@pytest.fixture
def trained_packet(tmp_path):
    training.train(TinyRuntime(), load_prompts(), tmp_path / "trained", dummy=True, **options(steps=2))
    return training.load_packet(tmp_path / "trained/response-adapter.pt", schema=training.EXPORT_SCHEMA)


def test_export_reloads_native_response_exactly_and_cannot_resume_training(trained_packet):
    runtime = TinyRuntime()
    game = training.build_game(runtime, trained_packet["config"])
    try:
        game.load_export_state_dict(trained_packet["adapter"])
        expected = game.evaluate(0, seed=42).cpu()
        assert expected.shape == (1, 3, 16, 16) and torch.isfinite(expected).all()
        with pytest.raises(RuntimeError, match="not a resumable training checkpoint"):
            game.step()
    finally:
        game.backend.adapter.close()
    restored_runtime = TinyRuntime()
    actual = response.evaluate(restored_runtime, trained_packet, index=0, seed=42)
    assert torch.equal(actual, expected)
    assert_no_hooks(restored_runtime)


@pytest.mark.parametrize("corruption", ["condition", "configuration", "source", "weights", "base"])
def test_export_rejects_changed_condition_config_source_and_weights_and_closes_hooks(
    trained_packet, corruption
):
    packet = deepcopy(trained_packet)
    runtime = TinyRuntime(seed=8 if corruption == "base" else 7)
    if corruption == "condition":
        packet["config"]["rows"][0]["positive"] += " under red moonlight"
    elif corruption == "configuration":
        packet["config"]["steps"] += 1
    elif corruption == "source":
        packet["adapter"]["config"]["implementation_sources"]["iris_sliders.adapters"] = "changed-source"
    elif corruption == "weights":
        first = next(iter(packet["adapter"]["network"].values()))
        first.flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="protocol differs|nonfinite"):
        response.evaluate(runtime, packet)
    assert_no_hooks(runtime)


@pytest.mark.parametrize("stage", ["prepare", "constructor"])
def test_build_failure_removes_all_native_adapter_hooks(monkeypatch, stage):
    runtime = TinyRuntime()

    def fail(*args, **kwargs):
        raise ValueError("intentional failure at " + stage)

    if stage == "prepare":
        monkeypatch.setattr(training.IrisResponseAdapter, "prepare", fail)
    else:
        monkeypatch.setattr(training, "ConditionalPG09", fail)
    with pytest.raises(ValueError, match="intentional failure"):
        training.build_game(runtime, training.configuration(load_prompts(), **options()))
    assert_no_hooks(runtime)


def test_invalid_update_budgets_reject_before_runtime_allocation(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid step budget reached runtime allocation")

    monkeypatch.setattr(training, "make_runtime", forbidden)
    for flag, value in (("--steps", "0"), ("--batch", "1")):
        with pytest.raises(ValueError, match="total steps|batch"):
            training.main(["--dummy", flag, value, "--outdir", str(tmp_path)])
    for stop in (0, 3, 1.5):
        with pytest.raises(ValueError, match="stop_after"):
            training.train(object(), load_prompts(), tmp_path, stop_after=stop, **options(steps=2))


def test_resume_requires_original_total_update_budget_before_binding_models(tmp_path):
    training.train(TinyRuntime(), load_prompts(), tmp_path, stop_after=1, dummy=True, **options(steps=2))
    with pytest.raises(ValueError, match="resume configuration differs"):
        training.train(
            object(), load_prompts(), tmp_path, resume=tmp_path / "training.pt", **options(steps=3)
        )


def test_cli_rejects_stop_budget_and_native_shape_before_runtime_allocation(monkeypatch, tmp_path):
    monkeypatch.setattr(
        training, "make_runtime", lambda args: pytest.fail("invalid config allocated a runtime")
    )
    for flags, match in (
        (["--dummy", "--steps", "2", "--stop-after", "3"], "stop_after"),
        (["--height", "513"], "multiples"),
    ):
        with pytest.raises(ValueError, match=match):
            training.main([*flags, "--outdir", str(tmp_path)])


def test_response_cli_rejects_fixture_mode_and_row_before_runtime_allocation(
    monkeypatch, tmp_path, trained_packet
):
    path = tmp_path / "adapter.pt"
    torch.save(trained_packet, path)
    monkeypatch.setattr(
        response, "make_runtime", lambda args: pytest.fail("invalid export allocated a runtime")
    )
    for flags, match in (([], "fixture mode"), (["--dummy", "--row", "999"], "out of bounds")):
        with pytest.raises(ValueError, match=match):
            response.main(["--adapter", str(path), "--out", str(tmp_path / "velocity.pt"), *flags])


def test_image_inference_rejects_raw_response_export_before_model_allocation(monkeypatch, tmp_path):
    import iris_sliders.infer as inference

    monkeypatch.setattr(
        inference, "IrisRuntime", lambda **kwargs: pytest.fail("unqualified render allocated model")
    )
    with pytest.raises(ValueError, match="rendering is unqualified"):
        inference.main(
            [
                "--adapter",
                str(tmp_path / "response-adapter.pt"),
                "--prompt",
                "fox",
                "--out",
                str(tmp_path / "image.png"),
            ]
        )


def test_negative_adam_second_moment_is_rejected_before_mutating_native_bank():
    runtime = TinyRuntime()
    game = training.build_game(runtime, training.configuration(load_prompts(), **options(preset="bispf")))
    try:
        game.step()
        before = game.state_dict()
        corrupted = deepcopy(before)
        for slots in corrupted["opt_g"]["state"].values():
            if "exp_avg_sq" in slots:
                slots["exp_avg_sq"].fill_(-1)
                break
        else:
            pytest.fail("original BISPF recipe did not expose Adam second-moment state")
        with pytest.raises(ValueError, match="second.moment|nonnegative|negative|exp_avg_sq"):
            game.load_state_dict(corrupted)
        assert equal_tree(before, game.state_dict())
    finally:
        game.backend.adapter.close()
    assert_no_hooks(runtime)


def test_native_positive_target_and_conditional_only_cfg_use_actual_bank():
    from iris3b.text.base import TextEncoding

    from iris_sliders.flow import guided_velocity

    runtime = TinyRuntime()
    game = training.build_game(runtime, training.configuration(load_prompts(), **options(steps=2)))
    try:
        selected = next(row for row in game.rows if 0 < float(row.row.timestep) < 1000)
        row = selected.row
        expected_target = guided_velocity(
            runtime.model,
            row.latent,
            row.timestep,
            TextEncoding(row.positive["embeds"], row.positive["mask"]),
            TextEncoding(row.negative["embeds"], row.negative["mask"]),
            cfg_scale=row.guidance,
        )
        assert torch.allclose(selected.target, expected_target, atol=2e-6, rtol=2e-6)
        game.step()
        game.step()
        conditional = game.backend.velocity(row.neutral, row.latent, row.timestep, scale=1.0)
        null = game.backend.velocity(row.negative, row.latent, row.timestep, scale=0.0)
        expected_student = null + row.guidance * (conditional - null)
        assert torch.equal(game.backend.generated(selected), expected_student)
        baseline = game.backend.velocity(row.neutral, row.latent, row.timestep, scale=0.0)
        assert not torch.equal(conditional, baseline)
        with (
            pytest.raises(ValueError, match="unipolar"),
            game.backend.adapter.active(game.prior.z, strength=-1),
        ):
            pass
    finally:
        game.backend.adapter.close()


@pytest.mark.parametrize("availability", ["missing", "wrong-version"])
def test_optional_hndl_unavailable_rejects_before_runtime_allocation(monkeypatch, tmp_path, availability):
    from importlib.metadata import PackageNotFoundError

    def unavailable(name):
        assert name == "hndl"
        if availability == "missing":
            raise PackageNotFoundError(name)
        return "0.0.0"

    monkeypatch.setattr(training, "version", unavailable)
    monkeypatch.setattr(
        training, "make_runtime", lambda args: pytest.fail("optional critic allocated native runtime")
    )
    with pytest.raises((ImportError, RuntimeError), match="hndl|HNDL"):
        training.main(["--critic", "hndl", "--outdir", str(tmp_path)])


def test_response_cli_optional_observer_executes_public_hypergan(tmp_path, trained_packet):
    pytest.importorskip("hypergan.diversity_metrics")
    path = tmp_path / "adapter.pt"
    torch.save(trained_packet, path)
    report = tmp_path / "diversity.json"
    response.main(
        [
            "--dummy",
            "--adapter",
            str(path),
            "--out",
            str(tmp_path / "velocity.pt"),
            "--diversity-report",
            str(report),
            "--latent-seeds",
            "13",
            "17",
        ]
    )
    result = json.loads(report.read_text())
    assert result["implementation"]["reviewed_commit"] == "a879c4fbff903ccefeba2194bcc7a13fe7b5abb7"
    assert result["observation"]["latent_seeds"] == [13, 17]
    assert result["observation"]["quality_validated"] is False
    assert next(iter(result["groups"].values()))["generated_count"] == 2


def test_response_observer_missing_dependency_rejects_before_runtime_allocation(
    monkeypatch, tmp_path, trained_packet
):
    from particle_sliders import hypergan_diagnostics

    path = tmp_path / "adapter.pt"
    torch.save(trained_packet, path)

    def unavailable(groups, **kwargs):
        raise hypergan_diagnostics.HyperGANDiagnosticsUnavailable("optional HyperGAN is unavailable")

    monkeypatch.setattr(hypergan_diagnostics, "response_diversity", unavailable)
    monkeypatch.setattr(
        response, "make_runtime", lambda args: pytest.fail("missing observer allocated native runtime")
    )
    with pytest.raises(hypergan_diagnostics.HyperGANDiagnosticsUnavailable):
        response.main(
            [
                "--dummy",
                "--adapter",
                str(path),
                "--out",
                str(tmp_path / "velocity.pt"),
                "--diversity-report",
                str(tmp_path / "report.json"),
            ]
        )
