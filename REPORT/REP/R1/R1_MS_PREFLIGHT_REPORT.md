# R1-MS0/1 Preflight — Drawer-only ManiSkill Transfer

Date: 2026-08-14

## Decision

The next experiment is **R1-MS0/1**, not a RoboTwin matrix and not a
three-task ManiSkill sweep.  The only initially unlocked task is
`OpenCabinetDrawer-v1`, using privileged `state_dict` observations and direct
instrumented excitation (Mode A).  Mode B, PushT, PegInsertion and all visual
observations remain locked.

This report is a protocol/readiness result.  It is **not** a physics-closure
result: ManiSkill and the drawer assets were absent at the first local probe,
so no simulator episode had been executed yet.

## Frozen scientific question

R1-MS tests whether the corrected R0.6 mechanism remains safe and
aggregate-useful when interaction complexity expands, without changing the
revision algorithm or adding perception error.  The independent variable is
the interaction distribution, not the method.

The frozen contract is serialized in
`runs/r1_ms/preflight/frozen_protocol.json`.  It preserves:

- the R0.6 tangent evidence and operator proposal;
- the dissipative-family hard feasibility constraint;
- equal-weight H2/H4/H8 utility acceptance with threshold zero;
- the history-aware fallback;
- the existing acceptance thresholds and proposal grammar;
- `H32` as evaluation-only information that is invisible to acceptance.

The protocol plus stage/gate manifest is content-addressed by SHA-256 in
`runs/r1_ms/preflight/summary.json`.  Any later mechanism change must produce a
new protocol hash and cannot be described as frozen R0.6 transfer.

## Ordered experiment

1. **MS0-A-C0** — fixed assets `1000`, `1040`, `1082`, each with the three
   development seeds, direct generalized-force excitation, and adequate
   object-specific physics. These are the first, middle and last IDs in the
   sorted Drawer registry and span 2, 1 and 3 target links respectively.
2. **MS0-A-C1** — only after C0 passes, inject the same controlled dissipative
   operator `-alpha*abs(qvel)*qvel` used to test R0.6 transfer.
3. **MS1-B** — robot/handle contact; locked until both Mode-A gates pass.
4. **PushT-v1** — locked until formal Drawer Mode A/B passes.
5. **PegInsertionSide-v1** — out-of-design stress test, locked until PushT.

This ordering separates object geometry/dynamics closure from controller,
contact and force-mapping errors.  If Mode A passes and Mode B fails, the
failure is localized to the embodied interface rather than the cabinet model.

C0 is a `3 assets x 3 seeds` matrix, not three randomly sampled episodes.
Closure is evaluated and reported separately for every asset; aggregation is
allowed only after all three assets individually pass. Each record must carry
joint friction, joint damping, drive stiffness/damping, joint type and limits,
the active DOF index, link mass/inertia, and gravity contribution. This
prevents cross-cabinet residuals of opposite sign from cancelling in an
aggregate statistic.

## State and force interface

The target drawer DOF uses the privileged local state

```text
[q, qvel, qacc, qf]
```

Current ManiSkill documentation exposes articulation `qpos`, `qvel` and `qf`
accessors.  The adapter therefore does not assume that `get_qacc()` exists:
when acceleration is not exposed, it computes `qacc` from adjacent `qvel`
snapshots and the actual simulator timestep.  This estimation path must be
recorded per sample.

The official Drawer source merges cabinets with heterogeneous DOF counts and
pads their articulation state.  Consequently, a fixed column is not a valid
cross-asset target-joint mapping.  The adapter gathers each environment using
`handle_link.joint.active_index`; a unit test uses different target indices in
the same batch to prevent regression to the single-asset assumption.

For multi-DOF generalization, residual power is implemented as

```text
r_tau^T qdot
```

on the last tensor dimension.  R1-MS0 itself remains a selected 1-DOF drawer
joint; PushT wrench/twist power is not activated early.

## Dynamics support mask

Every rollout sample is labeled with one of three states:

- `modeled`: inside the declared analytic dynamics support;
- `boundary`: near a joint limit or a contact-mode transition;
- `unsupported`: unmodeled collision, grasp constraint or task termination.

Boundary data are reported separately.  Unsupported data are not counted as
ordinary operator-selection failures.  This prevents later hybrid-contact
events from being silently folded into a binary validity flag.

## Frozen go/no-go gates

The stages expand only in this order:

1. C0 false revision `<1%` and no structured residual;
2. controlled C1 exact recovery `>90%`;
3. accepted stability `>=99%`;
4. native H16 gain `>0`, with no catastrophic H32 reversal;
5. acceptance rate `>20%`;
6. C2 forced wrong revision `<10%` and history-aware fallback better than the
   memoryless fallback.

The evaluator stops after the first failed or unavailable gate; later gates
are marked blocked rather than implicitly passed.

## Counterfactual contract

Every branch must restore the same simulator `state_dict` and make the
simulator execute the perturbed action/force.  Observational next states are
not reused as counterfactual labels.  Snapshots cover pre-contact, early
contact, sustained contact and near-transition phases; Mode A uses the same
branch schema even though its contact phases are initially inactive.

## Initial local readiness

The first probe found:

- existing SAPIEN `3.0.0b1`, Gymnasium `0.29.1`, Torch `2.4.1`;
- no installed `mani_skill` package;
- no local `OpenCabinetDrawer-v1` asset set;
- therefore `mode_a_smoke_ready=false` and
  `formal_matrix_unlocked=false`.

A project-local `.venv-maniskill` and `.maniskill` asset root are used so the
existing RoboTwin environment is not modified.  The official fixed ManiSkill
`3.0.1` package and only the Drawer task assets must pass integrity checks
before the first simulator smoke.

## Latest readiness and external blocker

The fixed official `v3.0.1` source tag (commit
`a4a4f9272ad64b1564035874b605ceb687b63ed8`) was installed into the isolated
environment.  The live probe now verifies:

- ManiSkill `3.0.1` and SAPIEN `3.0.3`;
- `OpenCabinetDrawer-v1`, `PushT-v1` and `PegInsertionSide-v1` registration;
- Drawer `max_episode_steps=100`;
- the exact 25 drawer asset IDs used by this package version.

No Drawer assets are present (`0/25`).  Both the official Python downloader
and a strict-TLS `curl` probe fail during the TLS handshake to
`storage1.ucsd.edu`; IPv4, HTTP/1.1, TLS1.2 and retries do not change the
failure.  A direct state-only CPU task construction reaches the expected
asset check and then fails at the same download endpoint.  Thus the current
blocker is specifically the upstream asset transport path, not task
registration, package import, GPU memory, or the APR-WM adapter.

A direct audit of all 25 registered `DataSource` objects found that every
Drawer archive has only an HTTPS URL on `storage1.ucsd.edu`; none has an
`hf_repo_id`, `github_url`, or published `checksum`. Thus ManiSkill 3.0.1
defines no official alternate source for this subset. An offline copy can
still preserve transport integrity by downloading from the registered URL on
a trusted machine/network, recording SHA256 there, and checking the same hash
after transfer, but that self-recorded hash is not upstream authenticity
metadata. The full audit is serialized in
`runs/r1_ms/preflight/asset_source_audit.json`.

The official asset group named `partnet_mobility_cabinet` includes both drawer
and door assets.  The project downloader deliberately derives only the 25
drawer IDs from `info_cabinet_drawer_train.json`.  Upstream does not publish
per-archive checksums for these sources, so the downloader records a local
tree SHA-256 after extraction but does not mislabel it as an upstream
authenticity checksum.  No third-party mirror or disabled TLS verification is
used.

Consequently:

```text
package_ready = true
task_registered = true
drawer_assets = 0/25
mode_a_smoke_ready = false
formal_matrix_unlocked = false
```

MS0-A-C0 remains the only next experimental action once the official assets
become reachable.  C1, Mode B and all later tasks remain blocked.

## Asset-free I/O smoke

While Drawer remains infrastructure-blocked, `PickCube-v1` was used only for
an `R1-MS0-I/O smoke`. Across seeds 8201/8211/8221, state snapshots restored
within `1.49e-8`, repeated nominal replay was bit-identical, all real simulator
counterfactual branches diverged from nominal, and HDF5 serialization read
back exactly. H2/H4/H8 are decision-plumbing horizons; H32 remains blind
evaluation-only. The three-state validity probe also preserved
`modeled/boundary/unsupported`.

This smoke explicitly serializes `scientific_result=false`,
`can_unlock_drawer_gate=false`, and `drawer_c0_evaluated=false`. It provides
no evidence about articulated closure, revision safety, or cross-asset
transfer. Its outputs are in `runs/r1_ms/io_smoke/`.

## Verification

The local protocol/adapter tests cover:

- selected-DOF oracle-state extraction;
- heterogeneous per-asset target-joint gathering;
- finite-difference acceleration fallback;
- multi-DOF generalized power and shape checking;
- three-state support classification;
- strict ordered-gate blocking;
- H32 blindness during acceptance;
- genuinely interventional counterfactual manifests.

At preflight creation these tests pass.  A simulator claim will be added only
after the exact ManiSkill package, task registration and Drawer assets are
verified locally.

## Upstream interface references

- ManiSkill installation: <https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html>
- OpenCabinetDrawer API/task card: <https://maniskill.readthedocs.io/en/latest/api/mani_skill/envs/tasks/mobile_manipulation/open_cabinet_drawer/index.html>
- state replay and contact/state APIs: <https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/advanced.html>
- articulation API: <https://maniskill.readthedocs.io/en/latest/api/mani_skill/utils/structs/articulation/index.html>
