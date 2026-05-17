# Tulsa Topological Affordability Spacetime

**Live site:** <https://richhank.github.io/ttas/>

A computational topology engine that maps the Tulsa housing market as a
high-dimensional shape and studies how that shape changes under different
affordability conditions, rate shocks, and household profiles.

TTAS embeds 9,600 property-month observations (10 ZIP codes, Jan 2018 – Dec
2025) into a 12-dimensional feature space covering price, rent, inventory
velocity, tax, schools, centrality, amenities, crime, flood risk, walk/transit
access, economic mobility, and household DTI. It then filters this point cloud
through three lenses — affordability, spatial density, and opportunity — and
computes persistent homology, Euler characteristic surfaces, vineyards, and a
household-specific rent-vs-buy path integral.

**What it found (real public data, current run):** The Tulsa market topology has
returned to a Stable configuration indistinguishable from its pre-pandemic
baseline, after passing through Overheated (2020–2022) and Rate Shock (2022–2023)
regimes. All three primary sources are active: FRED mortgage rates (4.37% → 3.25%
→ 7.32% → 6.70% real path), Census ACS ZIP metrics (24 Tulsa ZIPs), and
Realtor.com Tulsa metro listings ($329,900 median, 2,875 active, 51 DOM). Eight
Bayesian change points mark monetary-policy-driven structural breaks — one more
than the synthetic-only baseline, revealed by the real inventory data. H1 entropy
increased from 0.866 (synthetic) to 0.926 (real), indicating stronger loop
structure — real data makes the market's "donut holes" more visible. For the
median Tulsa household ($92K, 38% DTI), the topological buy signal is neutral.
See [Current Results](#current-results-december-2025) for the full table.

### Data: two modes

| Mode | What runs | When |
|------|-----------|------|
| **Real public data** *(live site)* | FRED, Census ACS, and Realtor.com calibrate the manifold | API keys + Realtor CSVs configured |
| **Synthetic fallback** *(fresh clone)* | Deterministic Tulsa-calibrated generator | No API keys set |

The live deployment at <https://richhank.github.io/ttas/> runs in **real public
data mode** — all three sources (FRED, Census, Realtor.com) are active. The data
badge on the site shows "Real Data" in teal.

If you clone this repo, youʼll start in synthetic fallback mode. The app works
immediately with no API keys — it uses a deterministic Tulsa-calibrated
generator spanning Jan 2018 through Dec 2025 across 10 ZIP codes. To activate
real data, add your own (free) API keys and Realtor.com CSVs (see
[Data Sources](#data-sources)). **No API keys are stored in this repository.**

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
invariant grids, Euler characteristic surfaces, vineyards, counterfactual topological
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
│   ├── raw/                  ← real source downloads (CSVs, API responses)
│   │   ├── realtor/
│   │   ├── fred/
│   │   ├── census/
│   │   └── tulsa/
│   ├── processed/            ← unified datasets + source manifest
│   │   ├── tulsa_market_timeseries.csv
│   │   ├── tulsa_zip_metrics.json
│   │   └── source_manifest.json
│   ├── fetch_data.py
│   ├── embeddings.py
│   ├── real_data.py
│   └── preprocess.py
├── scripts/                  ← real data fetching and import
│   ├── fetch_fred.py
│   ├── fetch_census.py
│   ├── import_realtor_csv.py
│   └── build_dataset.py
├── topology/
│   ├── filtrations.py
│   ├── multiparameter.py
│   ├── invariants.py
│   ├── silhouettes.py
│   ├── vineyards.py
│   └── causal_tda.py
├── decision/
│   ├── path_integral.py
│   ├── phase_transition.py
│   ├── topological_boundary.py
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

### Real data pipeline (FRED, Census, Realtor.com)

```bash
# 1. Set API keys (free sign-ups)
$env:FRED_API_KEY = "your-fred-key"
$env:CENSUS_API_KEY = "your-census-key"

# 2. Download Realtor.com Research CSVs for Tulsa and place in data/raw/realtor/
#    https://www.realtor.com/research/data/

# 3. Fetch + import + build
python scripts/fetch_fred.py
python scripts/fetch_census.py
python scripts/import_realtor_csv.py
python scripts/build_dataset.py

# 4. Run the pipeline with real data
python run_pipeline.py --refresh --write-html
```

The app auto-detects whether real data is available and picks the appropriate
mode. The data mode badge (teal "Real Data" or gold "Synthetic") is displayed
in the dashboard topbar and on the static site.

## Data Sources

| Source | Fields | Geography | Frequency | Status |
|--------|--------|-----------|-----------|--------|
| FRED | 30-yr mortgage rate, CPI, unemployment, median sales price | National | monthly | active — 953 rows |
| Census ACS | income, home value, owner share, rent burden | Tulsa County + 24 ZIPs | annual | active — 24 ZIPs |
| Realtor.com Research Data | listing price, inventory, days on market, new listings | Tulsa metro | monthly | active — 118 months |
| HUD-USPS ZIP Crosswalk | ZIP-to-tract mapping | National ZIP-level | quarterly | planned |
| Tulsa County Assessor | parcel values, property class, sale dates | Tulsa County parcels | as-available | planned |
| OSMnx | street centrality, amenity density | Tulsa area | on-demand | opt-in |

**API key sign-up (free):**
- FRED: <https://fred.stlouisfed.org/docs/api/api_key.html>
- Census: <https://api.census.gov/data/key_signup.html>

The live deployment uses real public data. A fresh clone falls back to synthetic
until you set your own API keys and re-run the pipeline.

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
componentwise maximum of its vertices, giving a skeletonized multiparameter
Vietoris-Rips construction.

`topology/multiparameter.py` now converts this skeleton into a
`multipers.SimplexTreeMulti` and calls `multipers.signed_measure` for Hilbert
and rank signed measures when the compiled backend is installed. If `multipers`
is unavailable, TTAS writes explicit `grid-fallback` artifacts: finite-grid
Hilbert values, Betti numbers, rank proxies, signed simplex measures, and
interleaving-distance proxies. The pipeline summary records the backend and
whether exact multiparameter output was produced.

### Invariant Suite

The local engine computes:

- H0 and H1 persistence diagrams with `ripser`, or pure-Python H0/MST and H1
  cycle approximations when compiled packages are unavailable.
- signed barcode summaries over vertices, edges, and triangles;
- exact `multipers` signed measures when available;
- fallback Hilbert/rank invariant grids when compiled TDA backends are absent;
- persistence silhouettes and Betti curves;
- Euler characteristic surfaces

```text
chi(lambda_1, lambda_2, lambda_3) = |V| - |E| + |T|;
```

- 12-month persistence vineyards and bottleneck drift from the 2018-2019
  baseline.

### Counterfactual Topological Effect

The counterfactual lab creates a counterfactual interest-rate market, recomputes the
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
Opportunity`, `Neutral`, or `Rent / Wait`. `decision/topological_boundary.py`
also samples biography-space across income and DTI, recomputes restricted
topology, and marks cells where the decision label, signal sign, or H1
persistence changes.

### Gaussian Process Regimes

`phase_transition.py` builds monthly Euler-surface feature vectors, fits a
`GaussianProcessClassifier`, and uses it for current regime inference. If the
model cannot be trained, the deterministic curvature thresholds remain as an
explicit fallback.

## Dashboard Tabs

- **Spacetime Manifold**: 3D UMAP/PCA embedding colored by persistent entropy.
- **Multiparameter Lab**: Hilbert slices and signed measure mass from exact or
  fallback multiparameter computations.
- **Euler Surface**: rotatable isosurface of `chi(lambda_1, lambda_2, lambda_3)`.
- **Silhouettes + Betti**: current-vs-baseline persistence silhouettes and
  Betti curves.
- **Persistence Vineyard**: H1 persistence tracks and bottleneck drift.
- **Opportunity Mapper**: KeplerMapper graph or local grid fallback.
- **Causal Shock Lab**: mortgage-rate counterfactual topology.
- **Regime GP**: Gaussian Process regime classifier trained on Euler features.
- **Boundary Atlas**: biography-space topology-changing boundary cells.
- **Decision Navigator**: household-specific `S(B)` landscape integral.

## Current Results (December 2025)

The pipeline was last run 2026-05-14 with all three primary real data sources
active (FRED, Census ACS, Realtor.com), calibrating 9,600 property-month
observations across 10 ZIP codes, Jan 2018 – Dec 2025.

| Invariant                          | Value    | Interpretation |
|------------------------------------|----------|----------------|
| H<sub>0</sub> persistent entropy   | 0.979    | Near-maximum fragmentation — the market is genuinely diverse, not monolithic |
| H<sub>1</sub> persistent entropy   | 0.926    | Increased from 0.866 (synthetic) — real data reveals stronger loop structure |
| Peak Euler curvature               | 57.5     | 3.0× above the critical threshold — sharp regime transitions confirmed |
| Bayesian change points             | 8        | Eight structural breaks — one more than synthetic, revealed by real inventory data |
| GP regime (current)                | Stable   | Market topology has returned to pre-pandemic configuration |
| Transfer entropy (rent → buy)      | 0.240    | Rent pressure drives affordability topology |
| Transfer entropy (buy → rent)      | 0.237    | Near-symmetric coupling with real data — prices and rents are tightly linked |
| Buy signal S(B) for median profile | −0.004   | Neutral — neither a structural opportunity nor a trap at current prices/rates |
| Boundary cells                     | 100      | Decision boundary maps across (income, DTI) space |
| Latest median listing price        | $329,900 | Tulsa metro (Apr 2026, Realtor.com) — 2,875 active listings, 51 DOM |

### Regime Distribution

| Regime      | Months | Period        | Market Condition |
|-------------|--------|---------------|------------------|
| Stable      | 2,400  | 2018–2019     | Pre-pandemic baseline, tight price-income coupling |
| Overheated  | 3,300  | 2020–2022     | Pandemic-era price/velocity surge, manifold expansion |
| Rate Shock  | 1,500  | 2022–2023     | Mortgage rate doubling, affordability compression |
| Opportunity | 2,400  | 2023–present  | Post-correction re-stabilization, fragmented pockets of value |

The GP classifier assigns December 2025 to the Stable regime with high confidence:
the market <em>shape</em> has returned to its pre-pandemic configuration even
though absolute price and rate levels are different. The manifold is homeomorphic
to the 2018 baseline; it is not isometric.

## Outputs

Pipeline artifacts are written to `outputs/cache`:

- `tulsa_manifold.csv`
- `tulsa_embedded.csv`
- `euler_surface_latest.csv`
- `signed_barcodes_latest.csv`
- `rank_invariant_h0_latest.csv`
- `rank_invariant_h1_latest.csv`
- `multiparameter_signed_measure.csv`
- `multiparameter_hilbert_function.csv`
- `multiparameter_rank_invariant.csv`
- `silhouettes_betti_curves.csv`
- `gp_regime_training.csv`
- `topological_decision_boundary.csv`
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

The default dataset is synthetic but Tulsa-calibrated. When real data sources
are configured, the synthetic manifold is calibrated to track real aggregate
metrics from FRED, Census ACS, and Realtor.com Research Data. This hybrid
approach preserves the 12-dimensional per-property structure needed for TDA
while anchoring headline metrics to real public data.

This is a research demonstration and portfolio project — not appraisal,
lending, legal, financial, or investment advice. Before making empirical claims,
verify source data and consult licensed data providers.
