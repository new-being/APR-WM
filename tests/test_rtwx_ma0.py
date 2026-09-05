"""MA0 conversion-audit helpers and CLI."""

from __future__ import annotations

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_ma0 import cup_gate, inferred_alignment_name


def test_cli_ma0():
    ns = build_parser().parse_args(["rtwx-ma0", "--p0-only"])
    assert ns.command == "rtwx-ma0"
    assert ns.p0_only is True


def test_alignment_name():
    assert inferred_alignment_name("next_index") == "next_state_as_action"
    assert inferred_alignment_name("same_index") == "same_index"


def test_cup_g0():
    assert cup_gate(4, 16, False) is True
    assert cup_gate(3, 16, False) is False
    assert cup_gate(1, 1, True) is True
