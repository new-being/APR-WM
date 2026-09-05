import json
from pathlib import Path

import pytest

from aprwm_v0.r1_rs1 import (
    FORMAL_SEEDS,
    PROBE_BANK,
    PROBES,
    REGIMES,
    R1RS1Config,
    _aggregate,
    _require_rs0_unlock,
    _torque_series,
)


def test_probe_bank_is_frozen_eight():
    assert tuple(PROBE_BANK) == PROBES
    assert len(PROBES) == 8
    assert all(PROBE_BANK[name].amplitude <= 0.12 for name in PROBES)


def test_formal_matrix_shape():
    assert len(FORMAL_SEEDS) == 5
    assert len(REGIMES) == 5
    assert len(FORMAL_SEEDS) * len(REGIMES) * len(PROBES) == 200


def test_rs1_locked_without_rs0(tmp_path: Path):
    with pytest.raises(RuntimeError, match="locked"):
        _require_rs0_unlock(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"rs0_passed": False}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="rs0_passed"):
        _require_rs0_unlock(bad)


def test_piecewise_probe_is_seed_deterministic():
    a = _torque_series(PROBE_BANK["P7"], 100, 0.002, seed=11)
    b = _torque_series(PROBE_BANK["P7"], 100, 0.002, seed=11)
    c = _torque_series(PROBE_BANK["P7"], 100, 0.002, seed=12)
    assert (a == b).all()
    assert not (a == c).all()


def test_aggregate_cneg_hard_gate():
    rows = []
    for seed in FORMAL_SEEDS:
        for regime in REGIMES:
            for probe in PROBES:
                rows.append(
                    {
                        "seed": seed,
                        "regime": regime,
                        "probe": probe,
                        "false_revision": False,
                        "exact_recovery": regime in {"C1-L", "C1-H"},
                        "coefficient_rel_err": 0.01,
                        "cneg_accepted_active": False,
                        "triggered": True,
                        "p_modeled": 1.0,
                        "c2_unknown": True,
                        "c2_wrong_explicit": False,
                        "accepted": regime in {"C1-L", "C1-H"},
                        "stable_h32": True,
                        "gain_h32": 0.05 if regime in {"C1-L", "C1-H"} else 0.0,
                    }
                )
    summary = _aggregate(rows, R1RS1Config())
    assert summary["rs1_go"] is True
    # One CNEG accepted active revision must fail the hard gate.
    for row in rows:
        if row["regime"] == "CNEG":
            row["cneg_accepted_active"] = True
            break
    failed = _aggregate(rows, R1RS1Config())
    assert failed["gates"]["cneg_accepted_active"]["pass"] is False
    assert failed["rs1_go"] is False
