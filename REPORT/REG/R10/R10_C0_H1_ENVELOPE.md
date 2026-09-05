# R10-C0-H1 Mechanical Envelope Worksheet

Date: 2026-08-17  
Status: **paper-design targets filled**; \(\tau_{\mathrm{FS}}\), \(\tau_{\mathrm{safe}}\)
**unfrozen**; **DEFERRED**; does not unlock C0  
Does not: freeze vendor SKU; treat \(I\) as CAD truth; fit \(\tau_f\) from data

Vertical-axis H1, \(\tau_g\simeq 0\). Numbers below are a **first paper
target**, not a measured envelope.

Prefer signal/noise via moderate \(\ddot q\), not high speed.
\(\tau_I=I\ddot q\approx 0.213\,\mathrm{N\cdot m}\) is already a usable
dynamic signal on a sub-Nm cell. Do not push \(\ddot q\) to tens of
\(\mathrm{rad/s}^2\).

| quantity | symbol | unit | paper target |
|---|---|---|---|
| arm / door length | \(L\) | m | \(0.40\) |
| total rotating mass | \(m\) | kg | \(1.0\) |
| shaft → COM | \(d_{\mathrm{COM}}\) | m | \(0.20\) (uniform-rod guess) |
| max angle (half-range) | \(q_{\max}\) | rad | \(0.50\) (\(\approx 28.6^\circ\)) |
| max speed | \(\dot q_{\max}\) | rad/s | \(1.0\) |
| max acceleration | \(\ddot q_{\max}\) | rad/s\(^2\) | \(4.0\) |
| inertia about shaft | \(I\) | kg·m\(^2\) | paper \(I_{\mathrm{arm}}=0.0533\) (slender rod); **CAD0 replaces by \(I_{\mathrm{CAD}}\)** |
| point-mass check | \(m d_{\mathrm{COM}}^2\) | kg·m\(^2\) | \(0.040\) (do **not** use as \(I\)) |
| friction budget | \(\tau_{f,\max}\) | N·m | \(0.10\) (conservative; **unconfirmed**) |
| peak torque | \(\tau_{\mathrm{peak}}\) | N·m | \(\approx 0.313\) |
| FS band (rule only) | \(1.5\sim 2\,\tau_{\mathrm{peak}}\) | N·m | \(\approx 0.47\sim 0.63\) → shop **\(0.5\)–\(1\,\mathrm{N\cdot m}\) class** |
| torque full scale | \(\tau_{\mathrm{FS}}\) | N·m | **unfrozen** (wait \(I_{\mathrm{CAD}}\), \(\tau_f\)) |
| safety software limit | \(\tau_{\mathrm{safe}}\) | N·m | **unfrozen** |

\[
I=\frac13 m L^2=\frac13(1.0)(0.40)^2=0.0533\,\mathrm{kg\cdot m^2},
\]

\[
\tau_{\mathrm{peak}}\approx I\ddot q_{\max}+\tau_{f,\max}
=0.0533\cdot 4+0.10\approx 0.313\,\mathrm{N\cdot m}.
\]

C0 excitation stays inside
\((q_{\max},\dot q_{\max},\ddot q_{\max})\). After CAD: replace \(I\) with **sensing-plane** \(I_{\mathrm{CAD}}\)
(`REPORT/REG/R10/R10_C0_H1_CAD0.md`), not a refined slender-rod formula.
After the rig exists: replace \(\tau_{f,\max}\). Then freeze
\(\tau_{\mathrm{FS}}\) and \(\tau_{\mathrm{safe}}\).

Next information: **deferred** (no hardware campaign now). See SIM-X
(`REPORT/REG/SIMX/SIMX_PREREG.md`). CAD0 solids wait until a real
sensing plane exists.
