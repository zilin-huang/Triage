# Triage Before Fine-Tuning

**Feasibility-Gated Selection and Monitored Adaptation of Embodied Foundation Models for a Fixed Physical Platform**

Zilin Huang, Zhengyang Wan, Zihao Sheng, Boyue Wang, Ruolin Yang, Yuhao Luo, Sikai Chen

University of Wisconsin-Madison

[Project page and videos](https://zilin-huang.github.io/Triage/) | Paper (arXiv, coming soon)

TRIAGE decides which pretrained driving or manipulation model a fixed platform can
run before any fine-tuning is spent on it. A sensing gate checks that the platform's
sensors carry what the model consumes, a latency gate checks that plans reach the
controller in time on the deployment hardware, the benchmark ranks the candidates that
pass both gates, and adaptation is monitored for the robustness it can cost.

This repository holds the code and result files that recompute tables and redraw
result figures of the paper, covering the simulation, real-log and manipulation
experiments and the five field experiments on a full-scale vehicle. Videos of the
closed-loop trips are on the [project page](https://zilin-huang.github.io/Triage/).

## Quick start

Requirements: Python 3.8 or later and NumPy; the two plotting scripts also need
Matplotlib. Run the scripts from the root of the repository. No arguments are needed.

```bash
python code/latency_gate.py
python code/selection_rules.py
python code/closed_loop_table.py
python code/paper_figures.py
python code/plot_trajectories.py
```

`expected_output.txt` holds the output of the first three scripts as run by the authors.

## Code (`code/`)

| Script | Reproduces | Data |
|---|---|---|
| `code/latency_gate.py` | Onboard columns of the latency table (Appendix A.5): median and 95th-percentile end-to-end latency of six candidates and the outcome of the 1.2 s gate; then the gate-restricted ranking of Stage 4, which selects DriveVLA-W0 with LTF as challenger | `data/latency/` |
| `code/selection_rules.py` | Selection-rule table (appendix): mean regret under leave-one-city-out for the top benchmark score without gates, the top benchmark score among candidates that pass both gates (Stage 4), a uniform pick, and the oracle, in the four deployment profiles | `data/selection/` |
| `code/paper_figures.py` | Plain versions of the result figures for the sensing gate (field of view against resolution), the latency gate (latency of 14 planners, closed loop under deadlines, stale plans against stale commands), the distribution gap (sensitivity of simulation-trained and real-data checkpoints, closed-loop rank changes), adaptation (fine-tuning dose curves and their monitors), and manipulation (latency and input-loss curves of five policies). Writes PDF and PNG files to `figures_out/` in about a minute | `data/results/` |
| `code/plot_trajectories.py` | Driven paths and lateral offsets of the closed-loop trips against the reference route, per configuration (`data/closed_loop/trajectories.png` is its output) | `data/closed_loop/traces/` |
| `code/closed_loop_table.py` | Upper block of the closed-loop table (Section 5): trips that reached the end, takeover positions, plan age and steering-wheel rate per configuration | `data/closed_loop/` |

## Data (`data/`)

- `data/latency/<candidate>.csv`: one row per plan produced with the vehicle parked and
  the full input path running. Columns: time the plan was ready, exposure times of the
  oldest and newest frame it used, inference time, number of frames, and whether the
  frames are simultaneous views. Times are in seconds from the start of the trip.
- `data/latency/gate_config.json`: the deadline, the number of warm-up plans dropped,
  the benchmark scores used for ranking, and the 95th-percentile latency of LTF on a
  hand-driven trip (the gate uses the larger of the parked and driving values).
- `data/selection/candidates.json`: PDMS of the nine configurations in each of the four
  NAVSIM cities and overall, each candidate's sensor contract and measured inference
  latency, and the four deployment profiles.
- `data/closed_loop/trips.csv`: one row per closed-loop trip of Field experiment V.
- `data/closed_loop/traces/`: one file per trip, sampled at 10 Hz while the vehicle
  drove autonomously: position and heading in a local metric frame (origin at the start
  of the reference route, y along its first 5 m), speed, distance along the route,
  lateral offset from the reference (positive to the right), steering-wheel angle, age
  of the plan being tracked, and whether that plan had expired. `index.csv` lists the
  trips and their outcomes; `reference_route.csv` is the reference route. The 22 trips
  of the closed-loop table are included, and the three trips of videos 3 to 5.
- `data/results/`: result files of the simulation, real-log and manipulation
  experiments, one folder per experiment, as read by `code/paper_figures.py`: closed-loop
  outcomes on Bench2Drive (`m4_matrix_v4/outcomes.jsonl`, one row per route, model,
  town, weather, seed and deadline), latency measurements, input-loss curves on NAVSIM
  and Bridge V2, probe readouts per model and city, and fine-tuning readouts. In-figure
  labels of the plain figures use working names: "information gate" is the sensing gate
  and "time gate" the latency gate.
- `data/field_experiments/`: result files of Field experiments I to V on the vehicle,
  with times relative to the start of each file. `I_latency/`: every timed call of
  five candidates on the onboard computer, alone, under full load, and without display
  tools. `II_probes/`: the plans and probe readings of each candidate on the local
  recording. `III_sensing_gate/`: per-window plan displacement, factory camera against
  the virtual view, and crop against resolution cut. `IV_latency_closed_loop/`: tracks
  in a local metric frame, engaged intervals, takeovers and their categories, speed,
  curvature and plan-age series, and the end-to-end latency split per plan.
  `V_all_stages/`: end-to-end latency of six candidates, LTF's latency while driving,
  and the Stage 3 probe readings on the local recordings.
- `code/vehicle/`: the code that produced these files, for reference (it needs the
  recordings and model weights, which are not included): `virtual_camera.py` reprojects
  the added 102 degree camera to the 64 degree view the candidates expect,
  `probe_rows.py` computes the probes, `sensing_gate_analysis.py` the crop, resolution
  and factory-camera comparisons, and `traj_io.py` the plan format.
- `data/results/revision_experiments/`: readouts of the experiments added in the
  revision: crop against downscale in closed loop and in open loop, TCP's crop residual
  over crop size, data size and seed, and ReCogDrive with its vision encoder adapting.

### Expected output

`expected_output.txt` holds the output of the three table scripts as run by the authors.

## Citation

```bibtex
@article{huang2026triage,
  title = {Triage Before Fine-Tuning: Feasibility-Gated Selection and Monitored Adaptation of Embodied Foundation Models for a Fixed Physical Platform},
  author = {Huang, Zilin and Wan, Zhengyang and Sheng, Zihao and
            Wang, Boyue and Yang, Ruolin and Luo, Yuhao and Chen, Sikai},
  year = {2026},
  journal = {arXiv preprint}
}
```

## Acknowledgments

This research was supported by the U.S. Department of Transportation (No. 69A3552348305)
and by grants from NVIDIA utilizing NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation
Edition GPUs.

## License

Released under the MIT License; see `LICENSE`.
