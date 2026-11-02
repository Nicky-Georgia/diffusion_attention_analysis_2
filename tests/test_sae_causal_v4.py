from pathlib import Path

import pandas as pd
import pytest
import torch

from diffusion_attention_analysis_v2.causal_controls import require_nonzero_dose
from diffusion_attention_analysis_v2.sae_fidelity_v4 import (
    TemporalTopKSAE, encode_at, group_file_split, load_research_sae,
    save_candidate, stable_split_hash,
)
from diffusion_attention_analysis_v2.sd3_counterfactual_v4 import (
    make_intervention_delta, new_heldout_manifest, sae_delta,
)


def test_holdout_split_groups_whole_samples():
    paths = [Path(f"/tmp/captures/sample{i:02d}/activations/step_{step:03d}/transformer_blocks__11.pt")
             for i in range(20) for step in (0, 13, 27)]
    train, val = group_file_split(paths, val_fraction=.25)
    assert len(train) == 45 and len(val) == 15
    assert not ({x.parent.parent.parent.name for x in train} &
                {x.parent.parent.parent.name for x in val})


def test_temporal_sae_roundtrip(tmp_path):
    model = TemporalTopKSAE(d_model=4, expansion_factor=2, k=2)
    x = torch.randn(7, 4)
    assert encode_at(model, x, 11).shape == (7, 8)
    p = tmp_path / "sae.pt"
    # Save helper intentionally records original d_model=1536; test custom reduced model
    # through the same checkpoint metadata and strict class loader.
    torch.save({"state_dict": model.state_dict(),
                "metadata": {"d_model": 4, "sae": {"architecture": "temporal", "k": 2,
                                                     "expansion_factor": 2}}}, p)
    new, meta = load_research_sae(p)
    assert torch.allclose(new(x, step=11)[0], model(x, step=11)[0], atol=1e-7)


def test_complete_sae_decomposition_and_nonzero_controls():
    from diffusion_attention_analysis_v2.sae import TopKSparseAutoencoder
    torch.manual_seed(31)
    sae = TopKSparseAutoencoder(d_model=4, expansion_factor=4, k=3)
    source = torch.randn(12, 4) + .5
    destination = torch.randn(12, 4) - .5
    f, h, e = sae_delta(sae, destination, source, step=13, ids=[0, 1, 2, 3])
    assert f.shape == h.shape == e.shape == source.shape
    assert torch.allclose(h+e,source-destination,atol=2e-6)
    sf = make_intervention_delta(sae,destination,source,mode="signflip_16",
                                 semantic_ids=[0,1,2,3],step=13,seed=11)
    sm = make_intervention_delta(sae,destination,source,mode="semantic_16",
                                 semantic_ids=[0,1,2,3],step=13,seed=11)
    assert sf.shape == sm.shape
    assert require_nonzero_dose(sf,sm,min_ratio=.999,max_ratio=1.001) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        require_nonzero_dose(torch.zeros_like(sm),sm)


def test_new_prompts_disjoint_and_frozen(tmp_path):
    prev = tmp_path / "old.csv"
    pd.DataFrame({"source_prompt":["old source"],"destination_prompt":["old destination"]}).to_csv(prev,index=False)
    a = new_heldout_manifest(prev,discovery_per_label=2,confirm_per_label=3,seed=9)
    b = new_heldout_manifest(prev,discovery_per_label=2,confirm_per_label=3,seed=9)
    pd.testing.assert_frame_equal(a,b)
    assert a.pair_id.is_unique
    assert len(a) == 16 * 5
    assert set(a.source_prompt).isdisjoint({"old source","old destination"})
    assert set(a.loc[a.split.eq("discovery"),"pair_id"]).isdisjoint(set(a.loc[a.split.eq("confirmatory"),"pair_id"]))
