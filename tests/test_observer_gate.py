import pytest
import torch
from iris_sliders.teacher import load_prompts

def test_missing_shared_observer_refuses_before_native_load(monkeypatch, tmp_path):
    from iris_sliders import hypergan_diagnostics, response, train as training

    def missing(*args, **kwargs):
        raise ModuleNotFoundError("hypergan.diversity_metrics")
    from types import SimpleNamespace
    monkeypatch.setattr(hypergan_diagnostics, "importlib", SimpleNamespace(import_module=missing))

    packet = {'schema': training.EXPORT_SCHEMA, 'config': training.configuration(load_prompts(), steps=2, height=16, width=16, trajectory_steps=4, records_per_pair=2), 'dummy': True}
    path = tmp_path / 'adapter.pt'
    torch.save(packet, path)
    monkeypatch.setattr(response, 'make_runtime', lambda *a: pytest.fail('missing observer loaded native model'))
    with pytest.raises(RuntimeError, match='Optional HyperGAN diversity diagnostics are unavailable'):
        response.main(['--dummy', '--adapter', str(path), '--out', str(tmp_path / 'v.pt'),
                       '--diversity-report', str(tmp_path / 'report.json')])
