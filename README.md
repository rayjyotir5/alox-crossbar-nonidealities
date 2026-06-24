# AlO\_x Memristor Crossbar Non-Idealities

Code and paper for **"From Device Physics to In-Situ Learning: A Unified
Simulation Study of AlO\_x Memristor Crossbar Non-Idealities"** (Jyotirmoy Ray,
Wipro Research). Compiled manuscript: [`paper.pdf`](paper.pdf).

A single, physically grounded AlO\_x valence-change device model is scaled,
without changing any equation, from one device to a 64x64 crossbar and a small
trained network, so that the three dominant non-idealities (device variability,
wire IR drop, and retention drift) can be compared on one footing. The headline
finding is that under this model they **separate by remedy**: variability is
correctable at write time (program-verify, 88.3 -> 96.5%), IR drop is absorbable
by hardware-in-the-loop training (63.5 -> 92.4%, versus 69.3% for output
calibration alone), and retention is only deferrable by rewriting. None of the
individual phenomena is new; the contribution is the unified, seed-averaged,
baselined comparison.

## Install
```bash
pip install -r requirements.txt   # numpy, scipy, scikit-learn, torch, matplotlib
```
Python 3.11. Everything runs in minutes on a laptop CPU.

## Reproduce the figures and numbers
| Script | Produces |
| --- | --- |
| `alox_crossbar.py` | Device I-V hysteresis, 64x64 variability, image program/read, analog MVM (Figs 1-3) |
| `crossbar_advanced.py` | MNA IR-drop solver, Arrhenius retention, conv-layer fidelity, digit-classifier accuracy (Figs A-C, IR table) |
| `crossbar_learn.py` | In-situ two-spiral learning, hardware-in-the-loop IR-drop training, post-training drift (Figs D-E) |
| `seeds_experiments.py` | Multi-seed error bars, mixed-state IR drop, E_a sensitivity, second task |
| `harden2.py` | In-situ training with decay schedule (multi-seed) + IR-recovery calibration baseline |
| `keshari_phase_energy.py` | Calibration to the published Ag2O device (Keshari et al. 2026), MNIST validation, operating-envelope phase diagrams, energy model |
| `aihwkit_baseline.py` | AIHWKit analog-inference baseline on MNIST (run in an AIHWKit venv); saves shared weights |
| `robotics_net.py` | PilotNet-style vision-based steering CNN + procedural driving data (inference-mapping workload) |
| `system_model.py` | Enhanced crossbar system model: im2col tiling, 1T1R access transistor, ADC/DAC quantization, read noise, calibrated readout |
| `sweep.py` / `analyze_envelope.py` / `robotics_demo.py` | Operating-envelope sweep, out-of-sample predictive validation, and the steering-inference demonstration figure |
| `crossbar_*_viz.py` / `filmstrips.py` | Animations + multi-frame figures: inference, IR-drop / drift dynamics, signal flow, parallel-read circuit |

```bash
python3 alox_crossbar.py
python3 crossbar_advanced.py
python3 crossbar_learn.py
python3 seeds_experiments.py
```
Random seeds are fixed; headline numbers are reported as mean +/- std over seeds.

## Model in brief
State variable `w in [0,1]` is filament completeness. Conduction blends an ohmic
LRS branch with a sinh-nonlinear HRS branch; the state switches only above SET /
RESET thresholds (origin of both hysteresis and nonvolatility). The crossbar
IR-drop is a full modified-nodal-analysis solve of the 2NM-node resistive
network. See the paper for equations, parameters, and limitations.

## Notes
This is a simulation study with an intentionally simplified model; it targets a
reproducible-study / workshop bar, not silicon-accurate prediction. The
manuscript appendix logs the full session provenance (human direction vs.
AI-assisted execution).

## Acknowledgement
Experiments and manuscript were produced with the assistance of Anthropic's
Claude (Claude Code). All results were inspected and verified by the author.
