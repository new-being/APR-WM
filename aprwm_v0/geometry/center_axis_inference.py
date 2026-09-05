"""Center–axis inference schedules (A0/A1/A2) — pure functions over frozen B2 + visibility LUT."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..rtwx_o0e0 import e_axis_deg
from .frozen_quotient_b2 import FrozenQuotientB2
from ..rtwx_o0e0 import TOP_KR
from .visibility_reference import AxialVisibilityReference

AXIS_CORRECT_DEG = 15.0
CONSISTENCY_WEIGHT = 0.0


@dataclass(frozen=True)
class VisObs:
    """Estimator-visible frame (no GT)."""

    fused_cloud: np.ndarray
    c_h: np.ndarray
    c_o: np.ndarray
    cam_h: np.ndarray
    cam_o: np.ndarray


@dataclass
class A0Result:
    p: np.ndarray
    n: np.ndarray
    d_ho: float
    meta: dict[str, Any]


@dataclass
class A1Result:
    n0: np.ndarray
    p0: np.ndarray
    n1: np.ndarray
    p1: np.ndarray
    n2: np.ndarray
    d_ho_0: float
    d_ho_1: float
    axis_error_n1_deg: float | None = None
    axis_error_n2_deg: float | None = None

    @property
    def p(self) -> np.ndarray:
        return self.p1

    @property
    def n(self) -> np.ndarray:
        return self.n2


@dataclass
class A2Result:
    p0: np.ndarray
    candidate_axes: list[np.ndarray]
    candidate_centers: list[np.ndarray]
    p_bar: np.ndarray
    n_final: np.ndarray
    center_dispersion_median: float
    center_dispersion_p90: float
    d_ho_final: float

    @property
    def p(self) -> np.ndarray:
        return self.p_bar

    @property
    def n(self) -> np.ndarray:
        return self.n_final


def visibility_center_per_view(
    axis_n: np.ndarray,
    cloud_centroid: np.ndarray,
    camera_center: np.ndarray,
    visibility_lut: AxialVisibilityReference,
) -> np.ndarray:
    return visibility_lut.estimate_center(axis_n, cloud_centroid, camera_center)


def visibility_center_fused(
    axis_n: np.ndarray,
    c_h: np.ndarray,
    c_o: np.ndarray,
    cam_h: np.ndarray,
    cam_o: np.ndarray,
    visibility_lut: AxialVisibilityReference,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    p_h = visibility_center_per_view(axis_n, c_h, cam_h, visibility_lut)
    p_o = visibility_center_per_view(axis_n, c_o, cam_o, visibility_lut)
    p = 0.5 * (p_h + p_o)
    d_ho = float(np.linalg.norm(p_h - p_o))
    return p, p_h, p_o, d_ho


def d_ho_at_axis(
    axis_n: np.ndarray,
    obs: VisObs,
    visibility_lut: AxialVisibilityReference,
) -> float:
    _, _, _, d = visibility_center_fused(
        axis_n, obs.c_h, obs.c_o, obs.cam_h, obs.cam_o, visibility_lut,
    )
    return d


def run_a0_joint(
    obs: VisObs,
    visibility_lut: AxialVisibilityReference,
    b2: FrozenQuotientB2,
) -> A0Result:
    assert CONSISTENCY_WEIGHT == 0.0

    def center_fn(n: np.ndarray) -> np.ndarray:
        p, _, _, _ = visibility_center_fused(
            n, obs.c_h, obs.c_o, obs.cam_h, obs.cam_o, visibility_lut,
        )
        return p

    res = b2.search_joint_center_fn(obs.fused_cloud, center_fn)
    p_hat = center_fn(res.axis)
    d_ho = d_ho_at_axis(res.axis, obs, visibility_lut)
    return A0Result(p=p_hat, n=res.axis, d_ho=d_ho, meta=res.meta)


def run_a1_center_first(
    obs: VisObs,
    n_const: np.ndarray,
    visibility_lut: AxialVisibilityReference,
    b2: FrozenQuotientB2,
    *,
    n_gt: np.ndarray | None = None,
) -> A1Result:
    n0 = np.asarray(n_const, dtype=np.float64).reshape(3)
    n0 = n0 / max(np.linalg.norm(n0), 1e-12)
    p0, _, _, d0 = visibility_center_fused(
        n0, obs.c_h, obs.c_o, obs.cam_h, obs.cam_o, visibility_lut,
    )
    res1 = b2.search_fixed_center(obs.fused_cloud, p0)
    n1 = res1.axis
    p1, _, _, d1 = visibility_center_fused(
        n1, obs.c_h, obs.c_o, obs.cam_h, obs.cam_o, visibility_lut,
    )
    res2 = b2.search_fixed_center(obs.fused_cloud, p1)
    n2 = res2.axis
    out = A1Result(
        n0=n0, p0=p0, n1=n1, p1=p1, n2=n2,
        d_ho_0=d0, d_ho_1=d1,
    )
    if n_gt is not None:
        out.axis_error_n1_deg = e_axis_deg(n1, n_gt)
        out.axis_error_n2_deg = e_axis_deg(n2, n_gt)
    return out


def run_a2_consensus(
    obs: VisObs,
    n_const: np.ndarray,
    visibility_lut: AxialVisibilityReference,
    b2: FrozenQuotientB2,
) -> A2Result:
    n0 = np.asarray(n_const, dtype=np.float64).reshape(3)
    n0 = n0 / max(np.linalg.norm(n0), 1e-12)
    p0, _, _, _ = visibility_center_fused(
        n0, obs.c_h, obs.c_o, obs.cam_h, obs.cam_o, visibility_lut,
    )
    cands = b2.coarse_candidates_fixed_center(obs.fused_cloud, p0, top_k=TOP_KR)
    assert len(cands) == TOP_KR
    axes = [c.axis for c in cands]
    centers = []
    for nk in axes:
        pk, _, _, _ = visibility_center_fused(
            nk, obs.c_h, obs.c_o, obs.cam_h, obs.cam_o, visibility_lut,
        )
        centers.append(pk)
    stack = np.stack(centers, axis=0)
    p_bar = np.median(stack, axis=0)
    dists = np.linalg.norm(stack - p_bar.reshape(1, 3), axis=1)
    res = b2.search_fixed_center(obs.fused_cloud, p_bar)
    d_ho = d_ho_at_axis(res.axis, obs, visibility_lut)
    return A2Result(
        p0=p0,
        candidate_axes=axes,
        candidate_centers=centers,
        p_bar=p_bar,
        n_final=res.axis,
        center_dispersion_median=float(np.median(dists)),
        center_dispersion_p90=float(np.percentile(dists, 90)),
        d_ho_final=d_ho,
    )


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    pos = labels == 1
    neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    tpr_list, fpr_list = [0.0], [0.0]
    tp = fp = 0
    n_pos = float(pos.sum())
    n_neg = float(neg.sum())
    for y in labels_sorted:
        if y == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)
    area = 0.0
    for i in range(1, len(tpr_list)):
        area += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) * 0.5
    return float(area)


def candidate_diagnostics(
    obs: VisObs,
    visibility_lut: AxialVisibilityReference,
    b2: FrozenQuotientB2,
    n_gt: np.ndarray,
    *,
    p_ref: np.ndarray | None = None,
) -> dict[str, Any]:
    """Per-frame candidate-level D_ho and joint scores (evaluator-only)."""
    n_gt = np.asarray(n_gt, dtype=np.float64).reshape(3)
    n_gt = n_gt / max(np.linalg.norm(n_gt), 1e-12)
    if p_ref is None:
        p_ref, _, _, _ = visibility_center_fused(
            n_gt, obs.c_h, obs.c_o, obs.cam_h, obs.cam_o, visibility_lut,
        )
    e_j, s_j, d_j = [], [], []
    for n in b2.sphere:
        n = np.asarray(n, dtype=np.float64)
        err = e_axis_deg(n, n_gt)
        p_vis, _, _, dho = visibility_center_fused(
            n, obs.c_h, obs.c_o, obs.cam_h, obs.cam_o, visibility_lut,
        )
        s_joint = b2.score_axis(obs.fused_cloud, p_vis, n)
        e_j.append(err)
        s_j.append(s_joint)
        d_j.append(dho)
    e_j = np.asarray(e_j, dtype=np.float64)
    s_j = np.asarray(s_j, dtype=np.float64)
    d_j = np.asarray(d_j, dtype=np.float64)
    y = (e_j <= AXIS_CORRECT_DEG).astype(np.float64)
    rho_s = float(np.corrcoef(s_j, e_j)[0, 1]) if np.isfinite(s_j).all() else float("nan")
    rho_d = float(np.corrcoef(d_j, e_j)[0, 1]) if np.isfinite(d_j).all() else float("nan")
    return {
        "e_axis_deg": e_j,
        "s_joint": s_j,
        "d_ho": d_j,
        "labels_correct": y,
        "auroc_neg_dho": _auroc(-d_j, y),
        "rho_s_joint_e_axis": rho_s,
        "rho_d_ho_e_axis": rho_d,
    }


def aggregate_candidate_diagnostics(per_frame: list[dict[str, Any]]) -> dict[str, float]:
    all_d, all_y = [], []
    rho_s, rho_d = [], []
    for pf in per_frame:
        all_d.extend(pf["d_ho"].tolist())
        all_y.extend(pf["labels_correct"].tolist())
        if np.isfinite(pf["rho_s_joint_e_axis"]):
            rho_s.append(pf["rho_s_joint_e_axis"])
        if np.isfinite(pf["rho_d_ho_e_axis"]):
            rho_d.append(pf["rho_d_ho_e_axis"])
    return {
        "auroc_neg_dho_agg": _auroc(-np.asarray(all_d), np.asarray(all_y)),
        "mean_rho_s_joint_e": float(np.nanmean(rho_s)) if rho_s else float("nan"),
        "mean_rho_d_ho_e": float(np.nanmean(rho_d)) if rho_d else float("nan"),
        "n_candidates_total": int(len(all_d)),
    }


def result_to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "__dataclass_fields__"):
        d = asdict(obj)
        for k, v in list(d.items()):
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif isinstance(v, list) and v and isinstance(v[0], np.ndarray):
                d[k] = [x.tolist() for x in v]
        return d
    return obj
