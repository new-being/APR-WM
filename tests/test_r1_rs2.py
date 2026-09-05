"""Unit tests for RS2-C0 contact closure plumbing."""

from __future__ import annotations

from aprwm_v0.r1_rs2 import (
    ADAPTER_SCALES,
    DEV_SEEDS,
    MODE_A_EXPOSURES,
    REPEATS,
    SCRIPTS,
    R1RS2C0Config,
    _aggregate_c0,
    _is_robot_door_pair,
    _manifest_sha256,
    _nearest_exposure_bin,
)


def test_rs2_c0_matrix_and_adapter_are_frozen():
    assert len(DEV_SEEDS) * len(SCRIPTS) * len(REPEATS) == 27
    assert DEV_SEEDS == (11001, 11011, 11021)
    assert SCRIPTS == ("slow_pull", "fast_pull", "pull_release")
    assert REPEATS == (0, 1, 2)
    assert ADAPTER_SCALES == (0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
    assert _manifest_sha256(R1RS2C0Config()) == _manifest_sha256(
        R1RS2C0Config()
    )


def test_robot_door_contact_filter_excludes_self_contacts():
    assert _is_robot_door_pair(
        "Door_handle", "gripper0_right_finger1_collision"
    )
    assert _is_robot_door_pair("robot0_link7_collision", "Door_panel")
    assert not _is_robot_door_pair(
        "gripper0_right_finger1_pad_collision",
        "gripper0_right_finger2_pad_collision",
    )
    assert not _is_robot_door_pair("Door_panel", "table_collision")


def test_exposure_bins_use_frozen_mode_a_references():
    for amplitude, exposure in MODE_A_EXPOSURES.items():
        assert _nearest_exposure_bin(exposure) == amplitude


def test_c0_aggregate_requires_every_gate():
    cfg = R1RS2C0Config()
    rows = []
    for seed in cfg.seeds:
        for script in cfg.scripts:
            for repeat in cfg.repeats:
                rows.append(
                    {
                        "base_seed": seed,
                        "script": script,
                        "repeat": repeat,
                        "finite": True,
                        "movement": 0.10,
                        "contact_active_fraction": 0.20,
                        "contact_p95": 0.10,
                        "contact_audit_nrmse": 1.0e-10,
                        "nrmse": 1.0e-4,
                        "active_nrmse": 1.0e-4,
                        "false_revision": False,
                    }
                )
    adapter = {
        "selected_scale": 1.25,
        "selected_relative_error": 0.10,
        "passed": True,
    }
    summary = _aggregate_c0(rows, adapter=adapter, config=cfg, smoke=False)
    assert summary["rs2_c0_pass"] is True
    rows[0]["false_revision"] = True
    failed = _aggregate_c0(rows, adapter=adapter, config=cfg, smoke=False)
    assert failed["rs2_c0_pass"] is False
    assert failed["classification"] == "infrastructure/interface_block"


def test_rs2_formal_seeds_are_held_out_and_adapter_frozen():
    from aprwm_v0.r1_rs2 import DEV_SEEDS, FROZEN_ADAPTER_SCALE, FROZEN_SCRIPT_AMP
    from aprwm_v0.r1_rs2_formal import FORMAL_SEEDS, REGIMES, R1RS2FormalConfig, _aggregate_formal

    assert FORMAL_SEEDS == (11101, 11111, 11121, 11131, 11141)
    assert set(FORMAL_SEEDS).isdisjoint(DEV_SEEDS)
    assert FROZEN_ADAPTER_SCALE == 2.0
    assert FROZEN_SCRIPT_AMP["slow_pull"] == 1.0
    assert FROZEN_SCRIPT_AMP["fast_pull"] == 1.5
    assert FROZEN_SCRIPT_AMP["pull_release"] == 1.0
    assert REGIMES == ("C0", "C1-L", "C1-H", "C2")
    assert len(R1RS2FormalConfig().seeds) * 4 * 3 * 3 == 180

    cfg = R1RS2FormalConfig()
    rows = []
    for seed in cfg.seeds:
        for regime in cfg.regimes:
            for script in cfg.scripts:
                for repeat in cfg.repeats:
                    accepted = regime == "C1-H" and script == "fast_pull" and repeat == 0
                    promoted = regime == "C1-L" and script == "slow_pull" and repeat == 0
                    probe_initial = regime in {"C1-L", "C1-H"} and script == "slow_pull"
                    rows.append(
                        {
                            "seed": seed,
                            "regime": regime,
                            "script": script,
                            "repeat": repeat,
                            "accepted": accepted,
                            "accepted_always": accepted,
                            "accepted_flipped_by_support": False,
                            "bucket_initial": (
                                "probe"
                                if probe_initial
                                else ("revise_worthy" if accepted else "tolerate")
                            ),
                            "bucket_final": "revise_worthy" if (accepted or promoted) else "tolerate",
                            "revise_worthy": accepted or promoted,
                            "promoted": promoted,
                            "voi_acquired": promoted,
                            "voi_V": 0.1 if promoted else 0.0,
                            "I_exit": 1.0 if accepted else 0.0,
                            "monitor_intensity": "elevated" if accepted else "none",
                            "gain_h32": 0.002 if accepted else 0.0,
                            "passivity_violation": False,
                        }
                    )
    summary = _aggregate_formal(rows, [], cfg)
    assert summary["gates"]["c0_specificity"]["pass"] is True
    assert summary["gates"]["tolerate_safety"]["pass"] is True
    assert summary["gates"]["passivity_among_installs"]["pass"] is True
    assert summary["gates"]["voi_path"]["pass"] is True
    assert summary["gates"]["monitor_consistency"]["pass"] is True
    assert summary["installs_support_exit_hard_filter"] is False
    rows[0]["accepted"] = True
    rows[0]["regime"] = "C0"
    rows[0]["monitor_intensity"] = "normal"
    rows[0]["I_exit"] = 0.0
    failed = _aggregate_formal(rows, [], cfg)
    assert failed["gates"]["c0_specificity"]["pass"] is False
