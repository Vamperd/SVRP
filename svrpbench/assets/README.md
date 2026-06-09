# SVRPBench — Assets

Figures, schematics, and the source paper for [SVRPBench](../README.md).

## Paper

- [SvrpBenchmark.pdf](SvrpBenchmark.pdf) — *SVRPBench: A Benchmark for
  Stochastic Vehicle Routing Problems*, Heakl, Shaaban, Takáč, Lahlou,
  Iklassov (2025). The reference write-up for the dataset, generation
  pipeline, and solver evaluation. **Cite this when using SVRPBench.**

## Schematic

| File | Description |
|------|-------------|
| [svrp.drawio.pdf](svrp.drawio.pdf) / [svrp.drawio.png](svrp.drawio.png) | End-to-end pipeline diagram: instance generation → travel-time / time-window sampling → solver → evaluation. |
| [svrp_logo.png](svrp_logo.png) | Project logo. |

## City visualizations

Real-world road networks compared to the synthetic instances generated for
each city:

| City     | Real network | Synthetic instance |
|----------|--------------|--------------------|
| Abu Dhabi | [abudhabi_real.png](abudhabi_real.png) | [abudhabi_synth.png](abudhabi_synth.png) |
| Michigan  | [michigen_real.png](michigen_real.png) | [michigen_synth.png](michigen_synth.png) |
| Milan     | [milan_real.png](milan_real.png) | [milan_synth.png](milan_synth.png) |
| (filtered roads) | [filtered_roads.png](filtered_roads.png) | — |

## Solver-comparison charts

PDF originals are kept for paper inclusion; PNGs are 600 DPI renders for
in-README embedding.

| Figure | Files |
|--------|-------|
| Per-solver scaling vs. instance size | [solver_scaling_by_size_v3.pdf](solver_scaling_by_size_v3.pdf) · [.png](solver_scaling_by_size_v3.png) (older: [v1 pdf](solver_scaling_by_size.pdf) · [v1 png](solver_scaling_by_size.png)) |
| Cost / runtime / feasibility bubble chart | [solver_performance_bubble_chart_v3.pdf](solver_performance_bubble_chart_v3.pdf) · [.png](solver_performance_bubble_chart_v3.png) (older: [v1 pdf](solver_performance_bubble_chart.pdf) · [v1 png](solver_performance_bubble_chart.png)) |
| Per-metric heatmaps across solvers | [solver_comparison_heatmaps.pdf](solver_comparison_heatmaps.pdf) · [.png](solver_comparison_heatmaps.png) |

## Per-instance solution renders

Sample solution drawings used in the paper figures:

| File | Solver | Size |
|------|--------|------|
| [instance_0_10_pomo.png](instance_0_10_pomo.png) | POMO | 10 |
| [instance_1_200_pomo.png](instance_1_200_pomo.png) | POMO | 200 |
| [instance_0_20_reinforce.png](instance_0_20_reinforce.png) | REINFORCE | 20 |

The full sweep lives under [visualizations/](visualizations/), one folder
per solver:

- [visualizations/aco/](visualizations/aco/) — ACO (CVRP, TWCVRP).
- [visualizations/nn2opt/](visualizations/nn2opt/) — NN + 2-opt.
- [visualizations/or_tools/](visualizations/or_tools/) — OR-Tools.
- [visualizations/attention/](visualizations/attention/) — Attention Model
  (10/20/50/100/200 customers, 10 instances each).
- [visualizations/pomo/](visualizations/pomo/) — POMO (same matrix).
- [visualizations/twvrp_10_single_depot/](visualizations/twvrp_10_single_depot/),
  [twvrp_20_single_depot/](visualizations/twvrp_20_single_depot/) —
  attention-model TWCVRP renders.

## Reproducing the PNG renders

The 600 DPI PNGs are produced from the PDFs with [`pdftoppm`](https://poppler.freedesktop.org/):

```bash
cd assets
for f in solver_*v3.pdf solver_comparison_heatmaps.pdf svrp.drawio.pdf; do
  pdftoppm -png -r 600 "$f" "${f%.pdf}" -singlefile
done
```
