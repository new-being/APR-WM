# R10-C0-H1 Reference Rig Architecture

Date: 2026-08-17  
Status: **FROZEN topology**; vendors and sensor FS **not** frozen;
**DEFERRED** until a real plant is available. Next work: SIM-X0.  
Depends on: `REPORT/REG/R10/R10_C0_ACQUISITION_DESIGN.md`  
Does not: unlock C0; freeze Interface/Beckhoff/Maxon SKUs; freeze
\(\tau_{\mathrm{FS}}\); gearbox; gravity-axis first rig

## Rig

\[
\boxed{\textbf{R10-C0-H1: vertical-axis 1-DoF direct-drive instrumented hinge}}
\]

\[
\text{servo}
\to
\text{flex coupling}
\to
\boxed{\text{inline rotary torque transducer}}
\to
\text{hinge shaft}
\to
\text{rigid arm / door}
\]

Hinge-side encoder on the **output shaft** (\(q_{\mathrm{hinge}}\)).
Motor encoder is servo/diagnostics only.

Vertical axis \(\Rightarrow\) C0 may take \(\tau_g(q)\simeq 0\).
\(I\) is **sensing-plane** \(I_{\mathrm{CAD}}\)
(`REPORT/REG/R10/R10_C0_H1_CAD0.md`), not motor+load lumped inertia:

\[
\tau_{\mathrm{meas}}=I_{\mathrm{CAD}}\ddot q+\tau_f(\dot q)+r_{\mathrm{real}}.
\]

No gearbox on H1. No contact (\(\tau_c\simeq 0\)). C1 may later add
payload (changes \(I\)) as a ground-truth mismatch.

## Roles

| module | H1 | why |
|---|---|---|
| body | vertical 1-DoF rigid arm/door | drop gravity as a C0 unknown |
| transmission | direct drive | no backlash / gear-efficiency residual |
| torque | **inline rotary** transducer | shaft torque, not housing reaction |
| science \(q\) | hinge-side encoder | \(q_{\mathrm{plant}}\), not \(q_{\mathrm{rotor}}\) |
| motor \(q\) | motor encoder → `diagnostics/` | compliance / torsion check |
| \(\tau\) ADC | DC-synced analog | same cycle as \(q\) |
| \(u\) | EtherCAT servo (e.g. CST class) | logged executed command |
| master | EtherCAT Distributed Clock | common cycle \(k\) |
| contact | none | C0 attribution |

\(\tau_{\mathrm{source}}=\) inline rotary transducer, not motor current,
not \(\tau_{\mathrm{cmd}}\). Reaction (fixed-housing) torque sensors are
not the H1 default: they mix motor-housing reaction with shaft torque.

Device **class** examples (not frozen SKUs): low-range rotary
transducers (e.g. Interface T11/T4 class), DC encoder slice
(EL5101-class), synced \(\pm10\,\mathrm{V}\) ADC (EL3702-class),
EtherCAT CST drive (EPOS4-class). Range and motor size wait on the
envelope below.

Full-scale rule (frozen, numbers not):

\[
\tau_{\mathrm{FS}}\gtrsim 1.5\sim 2\,\tau_{\mathrm{peak}}.
\]

Do not put a \(20\)–\(100\,\mathrm{Nm}\) cell on a \(0.1\,\mathrm{Nm}\)
rig.

## Clock

Prefer one DC domain so \(t_q[k]=t_\tau[k]=t_{\mathrm{bus}}[k]\).
Do **not** USB-poll encoder and torque on OS timestamps then interpolate.

\(u_{\mathrm{cmd}}[k]\) is command **into the drive**, not instantaneous
shaft torque. CST does **not** imply \(\Delta t_{\mathrm{actuator}}=0\).
Measure \(\Delta t_{u-\tau}\) from the inline transducer.

Schema: \(\Delta t_{q-\tau}\) (ideally \(\approx 0\) via DC);
\(\Delta t_{u-\tau}\) (actuator-chain property).

## Science vs servo encoders

`learner_visible/qpos` \(=q_{\mathrm{hinge}}\).  
`diagnostics/motor_qpos` \(=q_{\mathrm{rotor}}\).  
Coupling / shaft / mount compliance can make them differ even without
a gearbox.

## Next (still no purchase)

H1 **mechanical envelope** only — paper numbers:

1. arm/door length \(L\)
2. total mass \(m\)
3. shaft-to-COM distance
4. planned \(q_{\max},\dot q_{\max},\ddot q_{\max}\)
5. \(I_{\mathrm{CAD}}\) (load side of the transducer) and
   \(\tau_{\mathrm{peak}}\approx I_{\mathrm{CAD}}\ddot q_{\max}+\tau_{f,\max}\)

Then freeze \(\tau_{\mathrm{FS}}\), motor size, and safety torque limit.
Then procure.

Paper-design targets (not CAD, not C0): see
`REPORT/REG/R10/R10_C0_H1_ENVELOPE.md`.
Sensing-plane inertia cut: `REPORT/REG/R10/R10_C0_H1_CAD0.md`.
