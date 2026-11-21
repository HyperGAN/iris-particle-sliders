"""LR admission and exact shared recipe wiring; no pretrained/GPU work."""
import pytest
from iris_sliders.learning_rates import recipe_overrides

@pytest.mark.parametrize('key', ['lr', 'd_lr_mult', 'prior_lr_mult'])
@pytest.mark.parametrize('value', [True, False, 0, -1, float('nan'), float('inf'), '0.001'])
def test_invalid_rates(key, value):
    with pytest.raises(ValueError):
        recipe_overrides(**{key: value})


def test_optional_rates_keep_preset_defaults():
    assert recipe_overrides() == {}
    assert recipe_overrides(lr=2e-5, d_lr_mult=3) == {'lr': 2e-5, 'd_lr_mult': 3.0}


def test_configuration_and_cli_bind_all_rates():
    from iris_sliders.train import configuration, parser, build_game
    from iris_sliders.tiny import TinyRuntime
    rows = [{'neutral': 'a red cup', 'positive': 'a shiny red cup'},
            {'neutral': 'a blue bowl', 'positive': 'a shiny blue bowl'}]
    config = configuration(rows, steps=4, height=16, width=16, trajectory_steps=4,
                           records_per_pair=2, lr=2e-5, d_lr_mult=3, prior_lr_mult=4)
    assert configuration(**config) == config
    args = parser().parse_args(['--lr','0.00002','--d-lr-mult','3','--prior-lr-mult','4'])
    assert (args.lr,args.d_lr_mult,args.prior_lr_mult) == (2e-5,3,4)
    game = build_game(TinyRuntime(), config)
    try:
        assert (game.recipe.lr,game.recipe.d_lr_mult,game.recipe.prior_lr_mult) == (2e-5,3,4)
        assert sorted(set(game.base_lrs[0])) == pytest.approx([2e-5,8e-5])
        assert all(rate == pytest.approx(6e-5) for rate in game.base_lrs[1])
    finally:
        game.backend.adapter.close()
