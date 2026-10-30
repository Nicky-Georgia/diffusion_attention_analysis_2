import pytest
import torch

from diffusion_attention_analysis_v2.causal_controls import (
    active_nonsemantic_features,
    match_dose,
    require_nonzero_dose,
    signflip_spatial_control,
    weighted_latent_coverage,
)


def test_signflip_nonzero_and_preserves_each_token_norm():
    x = torch.arange(1, 49, dtype=torch.float32).reshape(8, 6)
    y = signflip_spatial_control(x, seed=13, step=11)
    assert torch.allclose(y.norm(dim=1), x.norm(dim=1))
    assert require_nonzero_dose(y, x) == pytest.approx(1.0)
    assert torch.equal(y, signflip_spatial_control(x, seed=13, step=11))


def test_zero_control_is_not_accepted_as_energy_matched():
    with pytest.raises(ValueError, match="Unmatched"):
        require_nonzero_dose(torch.zeros(8, 6), torch.ones(8, 6))
    with pytest.raises(ValueError, match="zero intervention"):
        match_dose(torch.zeros(8, 6), torch.ones(8, 6))


def test_nonsemantic_candidate_requires_actual_activity():
    source = torch.zeros(10, 5)
    dest = torch.zeros(10, 5)
    source[:, 0] = 5
    source[:, 2] = 3
    decoder = torch.ones(4, 5)
    assert active_nonsemantic_features(source, dest, decoder, forbidden=[0], count=1) == [2]
    with pytest.raises(ValueError, match="Only"):
        active_nonsemantic_features(source, dest, decoder, forbidden=[0], count=2)


def test_coverage_cannot_fabricate_effects_from_dead_features():
    source = torch.tensor([[1., 1., 0., 0.], [1., 1., 0., 0.]])
    dest = torch.zeros_like(source)
    decoder = torch.eye(4)
    cover = weighted_latent_coverage(source, dest, decoder, [0, 1, 2, 3], [1, 2, 4])
    assert cover[1] == pytest.approx(0.5)
    assert cover[2] == pytest.approx(1)
    assert cover[4] == pytest.approx(1)
