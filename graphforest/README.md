# graphforest

Directed graph-aware random forest for explainable fraud detection.

A practical middle path between tabular fraud models and GNNs — graph-aware random forests that banks can actually explain to regulators.

## Install

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from graphforest import DirectedGraphForestClassifier, GraphForestExplainer

model = DirectedGraphForestClassifier(
    source_col="sender_id",
    target_col="receiver_id",
    timestamp_col="txn_time",
    amount_col="amount",
)

model.fit(transactions_df, y)
preds = model.predict(transactions_df)
types = model.predict_fraud_type(transactions_df)

explainer = GraphForestExplainer(model)
explanation = explainer.explain(transactions_df, row_index=42)
```

## Fraud types

| Class | Type |
|-------|------|
| 0 | normal |
| 1 | friendly_fraud |
| 2 | identity_fraud |
| 3 | account_takeover |
| 4 | mule_ring |
| 5 | synthetic_identity |
| 6 | merchant_collusion |

## Graph edge schema
```
account → merchant       (payment flow)
device  → account        (access relationship)
email   → account        (identity link)
IP      → account        (access relationship)
account → beneficiary    (transfer flow)
card    → merchant       (card usage)
```
## Directed graph features

- `in_degree` / `out_degree`
- `weighted_in_degree` / `weighted_out_degree`
- `pagerank` — authority flows with money
- `hub_score` / `authority_score` — HITS algorithm
- `neighbor_fraud_rate` — fraction of flagged neighbors
- `cycle_score` — ring/carousel detection

## Roadmap

| Version | Focus |
|---------|-------|
| v0.1 | Directed graph feature forest (current) |
| v0.2 | Graph-aware split candidates inside the tree |
| v0.3 | Multi-class benchmarks on IEEE-CIS and Elliptic datasets |