# Tulsa Topological Affordability Spacetime

TTAS treats the Tulsa housing market as a non-stationary manifold rather than a
spreadsheet. Each property-month observation is embedded in a twelve-dimensional
feature lattice, filtered through affordability, density, and opportunity, and
summarized by persistent homology, Euler characteristic surfaces, causal shock
experiments, and a rent-vs-buy path integral.

Static portfolio site: <https://richhank.github.io/ttas/>

This repository stays fully local. It does not initialize Git, push to GitHub,
or require private API keys. The default data source is a deterministic
Tulsa-calibrated generator spanning January 2018 through December 2025. The code
is structured so public feeds from FRED, Zillow-style exports, Census,
OpenStreetMap, or local open-data files can replace synthetic columns later.

## Abstract

Housing affordability is a geometric phenomenon: income, credit constraints,
rent pressure, school quality, spatial centrality, amenities, risk, and market
velocity jointly determine whether a household sees a stable region, an
overheated boundary, or a rare pocket of opportunity. TTAS models this as a
time-indexed point cloud

```text
x_i(t) in R^12
```

and studies the topology of sublevel sets under the multiparameter filtration

```text
(lambda_1, lambda_2, lambda_3)
  = (affordability index, spatial density, opportunity score).
```

The engine computes persistence diagrams, signed barcode summaries, rank
invariant grids, Euler characteristic surfaces, vineyards, causal topological
effects of mortgage-rate shocks, and the path-integral buy signal

```text
S(B) = integral_0^infty Lambda_sub(t) - Lambda_full(t) dt.
```

Here `B` is the user's biography vector and `Lambda` denotes a persistence
landscape. A positive signal means the affordable sublevel set retains topology
not explained by the full market.

## Repository Layout

```text
ttas/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/
│   ├── fetch_data.py
│   ├── embeddings.py
│   └── preprocess.py
├── topology/
│   ├── filtrations.py
│   ├── invariants.py
│   ├── vineyards.py
│   └── causal_tda.py
├── decision/
│   ├── path_integral.py
│   ├── phase_transition.py
│   └── opportunity_mapper.py
├── dashboard/
│   ├── app.py
│   ├── assets/
│   └── callbacks/
├── visualizations/
│   ├── stills.py
│   └── plots.py
├── outputs/
├── README.md
└── run_pipeline.py
```

## Quick Start

```bash
cd C:/Users/Right/.codex/ttas
python run_pipeline.py --refresh --write-html
python dashboard/app.py
```

Open `http://127.0.0.1:8050`.

On Windows PowerShell:

```powershell
cd C:\Users\Right\.codex\ttas
python .\run_pipeline.py --refresh --write-html
python .\dashboard\app.py
```

## Docker

```bash
cd C:/Users/Right/.codex/ttas
docker compose -f docker/docker-compose.yml up --build
```

The Dockerfile uses Python 3.11 and installs the requested research stack:
`giotto-tda`, `multipers`, `ripser`, `persim`, `kmapper`, `plotly`, `dash`,
`umap-learn`, `pydiffmap`, `fredapi`, `osmnx`, `scikit-learn`, `econml`,
`dowhy`, and `tigramite`.

## Mathematical Core

### Feature Lattice

Each property-month is represented by twelve coordinates:

1. median listing price
2. rent-to-price ratio
3. inventory velocity
4. property tax rate
5. school rating
6. street centrality
7. amenity density
8. crime index
9. flood risk score
10. walk/transit score
11. economic mobility index
12. household DTI maximum

The generator encodes a pre-pandemic baseline, a pandemic-era overheating
phase, and a high-rate compression phase. Every latent shock is written to the
output frame so the model remains inspectable.

### Tri-Parameter Filtration

The filtration axes are:

```text
lambda_1 = 1 - ownership_cost / maximum_affordable_payment
lambda_2 = inverse local k-neighbor distance in R^12
lambda_3 = opportunity score from school, mobility, amenities, safety, and flood resilience
```

A vertex enters at its own lambda coordinate. An edge or triangle enters at the
componentwise maximum of its vertices, giving a finite approximation to a
skeletonized multiparameter Vietoris-Rips construction.

### Invariant Suite

The local engine computes:

- H0 and H1 persistence diagrams with `ripser`, or pure-Python H0/MST and H1
  cycle approximations when compiled packages are unavailable.
- signed barcode summaries over vertices, edges, and triangles;
- Hilbert/rank invariant grids;
- Euler characteristic surfaces

```text
chi(lambda_1, lambda_2, lambda_3) = |V| - |E| + |T|;
```

- 12-month persistence vineyards and bottleneck drift from the 2018-2019
  baseline.

### Causal Topological Effect

The causal lab creates a counterfactual interest-rate market, recomputes the
point cloud, and reports topological average treatment effects:

```text
Topological ATE_k = d_B(Dgm_k(factual), Dgm_k(counterfactual)).
```

When `persim` is present, this uses bottleneck distance. Otherwise the fallback
is a Hausdorff-style matching on finite persistence intervals.

### Decision Boundary

For a household biography `B`, TTAS restricts the market to affordable homes and
integrates the landscape difference:

```text
S(B) = integral_0^infty Lambda_sub(t) - Lambda_full(t) dt.
```

The dashboard normalizes the integral through `tanh` and maps it to `Buy
Opportunity`, `Neutral`, or `Rent / Wait`.

## Dashboard Tabs

- **Spacetime Manifold**: 3D UMAP/PCA embedding colored by persistent entropy.
- **Euler Surface**: rotatable isosurface of `chi(lambda_1, lambda_2, lambda_3)`.
- **Persistence Vineyard**: H1 persistence tracks and bottleneck drift.
- **Opportunity Mapper**: KeplerMapper graph or local grid fallback.
- **Causal Shock Lab**: mortgage-rate counterfactual topology.
- **Decision Navigator**: household-specific `S(B)` landscape integral.

## Outputs

Pipeline artifacts are written to `outputs/cache`:

- `tulsa_manifold.csv`
- `tulsa_embedded.csv`
- `euler_surface_latest.csv`
- `signed_barcodes_latest.csv`
- `rank_invariant_h0_latest.csv`
- `rank_invariant_h1_latest.csv`
- `vineyard_sequence.csv`
- `vineyard_tracks.csv`
- `topological_change_points.csv`
- `opportunity_mapper_nodes.csv`
- `pipeline_summary.json`

Portfolio HTML files are written to `outputs/figures` when `--write-html` is
used.

## References

- Gunnar Carlsson, "Topology and data", Bulletin of the American Mathematical
  Society, 2009.
- Afra Zomorodian and Gunnar Carlsson, "Computing persistent homology",
  Discrete and Computational Geometry, 2005.
- Steve Oudot, "Persistence Theory: From Quiver Representations to Data
  Analysis", 2015.
- Herbert Edelsbrunner and John Harer, "Computational Topology: An
  Introduction", 2010.

## Notes

The default dataset is synthetic but Tulsa-calibrated. It is designed for
portfolio-grade reproducibility and local experimentation, not appraisal,
lending, legal, or investment advice. Replace the generator columns with
licensed public or private data feeds before making empirical claims.
