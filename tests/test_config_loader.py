"""Tests for src.config_loader."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config_loader import (
    Config,
    DEFAULT_CONFIG_PATH,
    load_config,
)


def test_default_config_loads():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.source_path == DEFAULT_CONFIG_PATH
    # spot-check a few defaults
    assert cfg.chunking.child_target_chars == 400
    assert cfg.vector_db.hnsw_M == 16
    assert "1A" in cfg.cleaning.items_kept


def test_overrides_apply():
    cfg = load_config(overrides={"chunking": {"child_target_chars": 800}})
    assert cfg.chunking.child_target_chars == 800
    # other defaults untouched
    assert cfg.chunking.child_min_chars == 80


def test_overrides_deep_merge():
    cfg = load_config(
        overrides={"vector_db": {"hnsw_M": 32, "hnsw_search_ef": 80}}
    )
    assert cfg.vector_db.hnsw_M == 32
    assert cfg.vector_db.hnsw_search_ef == 80
    # untouched fields keep defaults
    assert cfg.vector_db.collection_name == "mag7_10k"


def test_validation_child_size_invariants():
    with pytest.raises(ValueError, match="child sizes"):
        load_config(
            overrides={
                "chunking": {
                    "child_min_chars": 500,
                    "child_target_chars": 400,  # < min, invalid
                }
            }
        )


def test_validation_kept_dropped_overlap():
    with pytest.raises(ValueError, match="overlap"):
        load_config(
            overrides={
                "cleaning": {
                    "items_kept": ["1", "1A"],
                    "items_dropped": ["1A", "1B"],  # 1A in both
                }
            }
        )


def test_validation_unknown_subheading_strategy():
    with pytest.raises(ValueError, match="subheading_strategies"):
        load_config(
            overrides={"cleaning": {"subheading_strategies": ["voodoo"]}}
        )


def test_validation_unknown_table_embedding_text():
    with pytest.raises(ValueError, match="table_embedding_text"):
        load_config(
            overrides={"chunking": {"table_embedding_text": "magic"}}
        )


def test_validation_hnsw_M_range():
    with pytest.raises(ValueError, match="hnsw_M"):
        load_config(overrides={"vector_db": {"hnsw_M": 100}})


def test_validation_temperature_range():
    with pytest.raises(ValueError, match="temperature"):
        load_config(overrides={"generation": {"llm_temperature": 5.0}})


def test_missing_section_errors():
    """A YAML file missing one of the required top-level sections should raise."""
    import tempfile, yaml as _yaml
    # Construct a config that has every section EXCEPT chunking
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        cfg = load_config()
        valid = {
            "cleaning": cfg.cleaning.__dict__,
            # chunking deliberately omitted
            "embedding": cfg.embedding.__dict__,
            "vector_db": cfg.vector_db.__dict__,
            "retrieval": cfg.retrieval.__dict__,
            "generation": cfg.generation.__dict__,
        }
        _yaml.dump(valid, f)
        tmp = f.name
    with pytest.raises(ValueError, match="missing required section"):
        load_config(tmp)
    Path(tmp).unlink()
