"""Score applicants: batch CSV or single JSON, with reason codes."""
import argparse
import json
import pickle

import numpy as np
import pandas as pd

TIERS = [(0.7, "APPROVE"), (0.4, "REVIEW"), (0.0, "DECLINE")]


def load_model():
    with open("artifacts/model.pkl", "rb") as f:
        art = pickle.load(f)
    return art["model"], art["features"], art["baseline"]


def tier(score: float) -> str:
    for cutoff, name in TIERS:
        if score >= cutoff:
            return name
    return "DECLINE"


def reason_codes(model, feats, baseline_proba, row: pd.Series) -> list[str]:
    """Per-applicant reason codes via one-feature-at-a-time attribution.

    For each feature we compare the model's probability with the feature
    held at its population median vs. the applicant's actual value.
    Positive delta = feature pushes toward approval (favorable).
    """
    base = pd.DataFrame([{f: row[f] for f in feats}])
    medians = pd.DataFrame([{f: row[f + "_median"] for f in feats}])
    deltas = []
    p_full = float(model.predict_proba(base)[0, 1])
    for f in feats:
        patched = base.copy()
        patched[f] = medians[f].iloc[0]
        p_patch = float(model.predict_proba(patched)[0, 1])
        deltas.append((f, p_full - p_patch))

    deltas.sort(key=lambda d: -abs(d[1]))
    out = []
    for f, d in deltas[:3]:
        if abs(d) < 0.02:
            continue
        direction = "+" if d < 0 else "-"  # raising default-risk -> unfavorable
        out.append(f"{f} {direction}")
    return out or ["no strong factors"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/applicants.csv")
    ap.add_argument("--json", dest="single", help="single applicant JSON")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    model, feats, baseline = load_model()

    if args.single:
        data = json.loads(args.single)
        row = {k: float(data[k]) for k in feats}
        for f in feats:
            row[f + "_median"] = float(baseline[f])
        r = pd.Series(row)
        proba = float(model.predict_proba(pd.DataFrame([{k: r[k] for k in feats}]))[0, 1])
        score = 1 - proba
        rc = reason_codes(model, feats, baseline, r)
        print(f"score: {score:.2f}  tier: {tier(score)}")
        print("reason codes:", ", ".join(rc))
        return

    df = pd.read_csv(args.path).head(args.limit)
    X = df[feats]
    probas = model.predict_proba(X)[:, 1]

    print(f"{'Applicant':<12}{'Score':>6}  {'Tier':<8}Top Reason Codes")
    for (_, r), p in zip(df.iterrows(), probas):
        score = 1 - p  # approval-worthiness score
        row = dict(r)
        for f in feats:
            row[f + "_median"] = float(baseline[f])
        rc = reason_codes(model, feats, baseline, pd.Series(row))
        print(f"{r['applicant_id']:<12}{score:>6.2f}  {tier(score):<8}{', '.join(rc)}")


if __name__ == "__main__":
    main()
