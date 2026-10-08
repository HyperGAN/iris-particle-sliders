"""Acquire the current shared stamp. No local game or regularizer."""

from particle_sliders import winning_formulation


def load_stamp(*, batch=None, g_lr=None):
    stamp = winning_formulation()
    declared = stamp.as_dict()
    if batch is not None:
        if batch < 1:
            raise ValueError("batch must be positive")
        declared["adv_batch"] = int(batch)
    if g_lr is not None:
        if g_lr <= 0:
            raise ValueError("generator learning rate must be positive")
        declared["g_lr"] = float(g_lr)
    stamp.require(declared)
    return stamp, declared


def record(stamp, declared):
    return {
        "architecture_id": stamp.architecture_id,
        "formulation_id": stamp.formulation_id,
        "provisional": bool(stamp.formulation_provisional),
        "declared": declared,
    }
