"""Regression tests for the statistical/logic core.

These cover:
  - F1: entry-point paths are anchored to the repo root, not the cwd, so a
    run from the repo root resolves the real data/ and output/ directories
    (previously hardcoded a 'hybrid-causal/' prefix that only worked when the
    cwd was the PARENT of the repo).
  - F2: the aggregate-data log-odds transform guards the zero-cell / full-cell
    edge cases instead of silently emitting log(0) = -inf or a
    divide-by-zero = +inf that would poison the downstream MCMC.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import model_hybrid  # noqa: E402
import ingest_data  # noqa: E402


# --- F1: path anchoring -------------------------------------------------

def test_model_root_points_at_repo_root():
    # ROOT is the parent of src/, i.e. the repo root itself.
    assert model_hybrid.ROOT == REPO_ROOT
    assert ingest_data.ROOT == REPO_ROOT
    # The real data file resolves from the anchored root regardless of cwd.
    assert (model_hybrid.ROOT / "data" / "hybrid_synthesis_input.json").exists()


def test_no_hardcoded_hybrid_causal_prefix_in_source():
    for name in ("model_hybrid.py", "ingest_data.py"):
        text = (REPO_ROOT / "src" / name).read_text(encoding="utf-8")
        assert "hybrid-causal/data" not in text
        assert "hybrid-causal/output" not in text


def test_main_raises_when_input_missing(tmp_path, monkeypatch):
    # Point ROOT at an empty dir: main() must fail loudly, not silently
    # return None with a print (the previous silent-failure sentinel).
    monkeypatch.setattr(model_hybrid, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        model_hybrid.main()


# --- F2: log-odds edge cases -------------------------------------------

def test_log_odds_valid_rows():
    df = pd.DataFrame({"events": [10, 20], "n": [100, 200]})
    out = model_hybrid.compute_ad_log_odds(df)
    expected = np.log(np.array([10, 20]) / np.array([90, 180]))
    assert np.allclose(out, expected)
    assert np.all(np.isfinite(out))


def test_log_odds_zero_events_raises():
    # events == 0 would give log(0) = -inf; must raise instead.
    df = pd.DataFrame({"events": [0, 20], "n": [100, 200]})
    with pytest.raises(ValueError):
        model_hybrid.compute_ad_log_odds(df)


def test_log_odds_full_events_raises():
    # events == n would give divide-by-zero = +inf; must raise instead.
    df = pd.DataFrame({"events": [100, 20], "n": [100, 200]})
    with pytest.raises(ValueError):
        model_hybrid.compute_ad_log_odds(df)


def test_log_odds_empty_raises():
    df = pd.DataFrame({"events": [], "n": []})
    with pytest.raises(ValueError):
        model_hybrid.compute_ad_log_odds(df)
