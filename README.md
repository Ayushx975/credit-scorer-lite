# Credit Scorer Lite

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-FC6600?style=flat) ![License MIT](https://img.shields.io/badge/License-MIT-green.svg)

Gradient-boosted credit scoring with **reason codes** — every score comes with explainable factors (like a real CIBIL-style report).

Built to explore fair, transparent lending models for the Indian retail-credit context: small training footprint, no PII, and human-readable decline/ approval reasons.

## Features

- Synthetic applicant dataset (income, obligations, history, utilization, inquiries)
- XGBoost classifier with 5-fold cross-validated AUC reporting
- **Reason codes**: per-applicant one-feature-at-a-time attributions vs. population median (e.g. "utilization -", "income +")
- Risk tiers: APPROVE / REVIEW / DECLINE with cutoffs tuned via precision-recall trade-off
- CLI batch scoring + single-applicant scoring from JSON

## Quick Start

```bash
pip install -r requirements.txt
python train.py                # trains + evaluates, saves model artifacts
python score.py data/applicants.csv          # batch score
python score.py --json '{"income": 65000, "obligations": 18000, "history_months": 14, "utilization": 0.72, "inquiries_6m": 4, "age": 27}'
```

## Sample Output

```
Applicant  Score  Tier     Top Reason Codes
APP-0002    0.42   REVIEW   history_months -, income -, utilization -
APP-0114    0.07   DECLINE  history_months -, utilization -, income -
APP-0207    0.97   APPROVE  utilization -
```

5-fold CV AUC: **~0.81** on the synthetic benchmark (realistic — credit data is noisy by nature; the value here is the pipeline + reason codes, not a Kaggle-topping score).

## Why Reason Codes Matter

Regulators (RBI's Fair Practices Code) require lenders to communicate decline reasons. Pure ML scores are not enough — this repo shows a minimal pattern for attaching signed feature attributions to every decision.

## Tech

Python 3.11 | XGBoost | scikit-learn | pandas

## Related

- MuleShield — mule account & suspicious transaction detection (https://github.com/Ayushx975/muleshield-web)
- upi-anomaly-guard — UPI anomaly detection (https://github.com/Ayushx975/upi-anomaly-guard)

## License

MIT
