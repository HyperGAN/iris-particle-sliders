"""Consumer regression for the frozen shared optimizer restore repair."""

from copy import deepcopy

import pytest
from test_pg09_training import equal_tree

from iris_sliders.teacher import load_prompts
from iris_sliders.tiny import TinyRuntime
from iris_sliders.train import build_game, configuration


@pytest.mark.parametrize("optimizer", ["opt_g", "opt_d"])
@pytest.mark.parametrize(
    "corruption",
    [
        "negative",
        "missing-moment",
        "cleared-slots",
        "deleted-entry",
        "cleared-history",
        "missing-history",
        "extra-entry",
        "zero-used-clock",
    ],
)
def test_started_history_rejects_atomically_on_actual_native_iris(optimizer, corruption):
    game = build_game(
        TinyRuntime(),
        configuration(
            load_prompts(),
            steps=2,
            height=16,
            width=16,
            trajectory_steps=4,
            records_per_pair=2,
            preset="bispf",
        ),
    )
    try:
        game.step()
        before = game.state_dict()
        changed = deepcopy(before)
        packet = changed[optimizer]
        key, slots = next(iter(packet["state"].items()))
        if corruption == "negative":
            slots["exp_avg_sq"].fill_(-1)
        elif corruption == "missing-moment":
            del slots["exp_avg_sq"]
        elif corruption == "cleared-slots":
            slots.clear()
        elif corruption == "deleted-entry":
            del packet["state"][key]
        elif corruption == "cleared-history":
            packet["state"].clear()
        elif corruption == "missing-history":
            del packet["state"]
        elif corruption == "zero-used-clock":
            slots["step"].zero_()
        else:
            packet["state"][1000000] = deepcopy(slots)
        with pytest.raises(ValueError, match="optimizer|Adam"):
            game.load_state_dict(changed)
        assert equal_tree(before, game.state_dict())
    finally:
        game.backend.adapter.close()


def test_unstarted_native_game_accepts_its_honest_empty_optimizer_history():
    game = build_game(
        TinyRuntime(),
        configuration(load_prompts(), steps=2, height=16, width=16, trajectory_steps=4, records_per_pair=2),
    )
    try:
        before = game.state_dict()
        game.load_state_dict(before)
        assert equal_tree(before, game.state_dict())
        assert game.step()["step"] == 1
    finally:
        game.backend.adapter.close()
