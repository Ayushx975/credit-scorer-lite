"""Train gradient-boosted credit model, report CV AUC, persist artifacts."""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

FEATURES = ["income", "obligations", "history_months", "utilization", "inquiries_6m", "age"]
ART = "artifacts"
os.makedirs(ART, exist_ok=True)


def main() -> None:
    df = pd.read_csv("data/applicants.csv")
    X, y = df[FEATURES], df["default"]

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, eval_metric="auc",
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"5-fold CV AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    model.fit(X, y)
    baseline = {f: float(X[f].median()) for f in FEATURES}
    with open(os.path.join(ART, "model.pkl"), "wb") as f:
        pickle.dump({"model": model, "features": FEATURES, "baseline": baseline}, f)
    print("saved artifacts/model.pkl")


if __name__ == "__main__":
    main()
