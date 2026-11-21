"""Use the exact shared gate without replacing Iris's source-bound core."""

import sys
from dataclasses import asdict
from hashlib import sha256
from threading import Lock
from types import ModuleType
from zipfile import ZipFile

import torch

CORE_REVISION = "b1f780decc516808cfd3019d5e5b674457161f86"
MODULE_PATH = "particle_sliders/adaptive_steering.py"
MODULE_SHA256 = "5fc2a5256a935eca9cecc365d9ac62e5e94cdc375d2ab33c5dc26eb344241814"
_MODULE_NAME = f"{__name__}._core_{CORE_REVISION}"
_LOAD_LOCK = Lock()


def _shared_gate(core_wheel):
    # The installed core stays pinned for strict RAW and FULL packet admission.
    with ZipFile(core_wheel) as wheel:
        try:
            source = wheel.read(MODULE_PATH)
        except KeyError as error:
            raise ValueError("Iris needs a core wheel containing the shared steering module") from error
    if sha256(source).hexdigest() != MODULE_SHA256:
        raise ValueError("Iris needs the exact shared steering module from core PR #269")
    with _LOAD_LOCK:
        if _MODULE_NAME not in sys.modules:
            module = ModuleType(_MODULE_NAME)
            sys.modules[_MODULE_NAME] = module
            try:
                exec(compile(source, f"{core_wheel}/{MODULE_PATH}", "exec"), module.__dict__)  # noqa: S102
            except BaseException:
                del sys.modules[_MODULE_NAME]
                raise
        return sys.modules[_MODULE_NAME]


class LatentGatePlan:
    """Explicit calibrated channel readout and request-local CFP or CLABS."""

    def __init__(
        self, *, core_wheel, method, parameter, calibration_sha256, direction=None, seed=None, clabs=None
    ):
        if (
            not isinstance(calibration_sha256, str)
            or len(calibration_sha256) != 64
            or any(character not in "0123456789abcdef" for character in calibration_sha256)
        ):
            raise ValueError("Provide the SHA-256 of the Iris calibration record")
        if type(parameter) not in (int, float):
            raise ValueError("Provide a numerical Iris gate threshold, bias, or random rate")
        self._shared = _shared_gate(core_wheel)
        self._method, self._parameter, self._seed = method, float(parameter), seed
        self._clabs = self._shared.CLABSParameters(**clabs) if clabs is not None else None
        self._calibration_sha256 = calibration_sha256
        self._direction = None
        if method == "random":
            if direction is not None:
                raise ValueError("The matched random Iris control does not use a direction")
        else:
            if direction is None:
                raise ValueError("Iris CFP and CLABS require a calibrated solver-channel direction")
            probe = torch.as_tensor(direction, dtype=torch.float32).detach().clone()
            if (
                probe.ndim != 1
                or not torch.isfinite(probe).all()
                or not torch.isclose(probe.norm(), probe.new_tensor(1.0))
            ):
                raise ValueError("Iris direction must be a finite unit vector")
            self._direction = probe
        self._new_gate()

    def _new_gate(self):
        return self._shared.ActivationGate(
            self._method,
            self._parameter,
            seed=self._seed,
            clabs=self._clabs,
        )

    def _score(self, latent, strength):
        if self._direction is None:
            return None
        channels = latent[0].mean(dim=(1, 2), dtype=torch.float32)
        with torch.autocast(latent.device.type, enabled=False):
            return float(self._shared.activation_projection(channels, self._direction, strength=strength))

    def _receipt(self):
        return {
            "core_revision": CORE_REVISION,
            "core_module_sha256": MODULE_SHA256,
            "signal": "FP32 spatial means of the pre-edit Iris solver input channels",
            "calibration_sha256": self._calibration_sha256,
            "method": self._method,
            "parameter": self._parameter,
            "seed": self._seed,
            "clabs": asdict(self._clabs) if self._clabs is not None else None,
            "direction": self._direction.tolist() if self._direction is not None else None,
        }
