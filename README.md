# Player Segmentation — RFM-Style Engagement & Monetization Analysis

Segments "players" into behavioral tiers using **Recency, Frequency, and
Monetary value (RFM)** — the classic engagement/monetization framework used
across games, apps, and e-commerce — combined with **K-Means clustering** as
an unsupervised cross-check.

## Why this dataset

Proprietary game session/telemetry logs aren't publicly available, so this
project uses the real **UCI "Online Retail" dataset** (541,909 invoice
line-items, ~4,300 customers, Dec 2010–Dec 2011, UK online retailer). RFM is
a domain-agnostic technique — it maps directly onto a player-engagement
vocabulary (customer → player, invoice → monetized session, days-since-last-
purchase → days-since-last-login), and the same pipeline runs unchanged on a
real session/IAP log if one is substituted in. This trade-off, and its
limits, are documented explicitly inside the notebook (Section 10).

## What's inside

```
player-segmentation/
├── notebooks/
│   ├── Player_Segmentation.ipynb   ← main deliverable, fully executed
│   └── player_segmentation.py      ← same analysis as a plain script (jupytext)
├── data/
│   ├── online_retail.csv           ← raw source data
│   └── players_segmented.csv       ← output: every player + scores + segment
├── figures/                        ← all charts, exported as PNG
└── README.md
```

## Method

1. **Clean** — drop rows with no player id, cancelled orders, non-positive
   quantity/price (removed ~26% of rows, matching known data-quality issues
   in this well-studied dataset).
2. **Engineer features** — per player: `Recency` (days since last activity),
   `Frequency` (distinct engagement events), `Monetary` (lifetime spend).
3. **Rule-based segmentation** — quintile-score each of R/F/M (1–5) and map
   the combined score to 10 named segments (Champions, Loyal Players,
   Can't Lose (High-Value), Hibernating/Churned, etc.) using a standard
   RFM segment map.
4. **Unsupervised segmentation** — log-transform + standardize R/F/M, then
   K-Means (k chosen via elbow + silhouette analysis) into 4 clusters,
   profiled and named from their centroids.
5. **Compare** the two approaches via a cross-tab/heatmap to sanity-check
   that an independent method recovers similar structure.
6. **Business read-out** — who to retain, who to win back, who's cheap to
   let go — plus an explicit limitations section.

## Headline result

- **15%** of players ("Champions") drive **~51%** of total revenue — a
  Pareto-style concentration typical of both retail and game monetization.
- K-Means independently recovers the same top/bottom tiers as the rule-based
  segments (see the cross-tab heatmap), validating the rule-based approach.
- The full per-player segment assignment is exported to
  `data/players_segmented.csv` for downstream use (targeting, dashboards,
  CRM export).

## Tools

Python · pandas · scikit-learn (KMeans, StandardScaler, silhouette_score) ·
matplotlib · seaborn · Jupyter

## Run it yourself

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter nbconvert --to notebook --execute --inplace notebooks/Player_Segmentation.ipynb
```

## Source

UCI Machine Learning Repository — "Online Retail" Data Set, Dr Daqing Chen,
School of Engineering, London South Bank University.
