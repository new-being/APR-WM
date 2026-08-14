import json

import pytest

from aprwm_v0.config import load_config


def test_partial_config_uses_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"train": {"steps": 2}}), encoding="utf-8")
    config = load_config(path)
    assert config.train.steps == 2
    assert config.train.n_objects == 6


def test_unknown_config_key_fails(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"train": {"not_a_field": 2}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown keys"):
        load_config(path)

