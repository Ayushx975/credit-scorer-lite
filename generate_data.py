"""Synthetic applicant data generator for credit scoring."""
import csv
import os
import random

random.seed(7)

N = 1500
FIELDS = ["applicant_id", "income", "obligations", "history_months", "utilization", "inquiries_6m", "age", "default"]


def default_prob(inc: float, obl: float, hist: int, util: float, inq: int) -> float:
    dti = obl / max(inc, 1.0)
    p = 0.02
    p += 0.65 * max(0.0, dti - 0.35) ** 1.3
    p += 0.30 * max(0.0, util - 0.45) ** 1.1
    p += 0.06 * min(inq, 6) / 6.0
    if hist < 12:
        p += 0.15
    elif hist > 60:
        p -= 0.03
    return min(max(p, 0.0), 0.95)


def main() -> None:
    os.makedirs("data", exist_ok=True)
    rows = []
    for i in range(N):
        inc = random.gauss(65000, 30000) if random.random() < 0.85 else random.uniform(200000, 500000)
        inc = max(inc, 15000)
        # defaulters: higher DTI + utilization correlated cluster
        is_def = random.random() < 0.24
        if is_def:
            obl = inc * random.uniform(0.30, 0.60)
            util = random.uniform(0.55, 1.0)
            hist = random.randint(0, 24)
            inq = random.randint(2, 8)
        else:
            obl = inc * random.uniform(0.05, 0.32)
            util = random.uniform(0.0, 0.55)
            hist = random.randint(6, 180)
            inq = random.randint(0, 3)
        age = random.randint(21, 60)
        p = default_prob(inc, obl, hist, util, inq)
        d = 1 if random.random() < p else 0
        rows.append({
            "applicant_id": f"APP-{i:04d}",
            "income": round(inc, 2),
            "obligations": round(obl, 2),
            "history_months": hist,
            "utilization": round(util, 3),
            "inquiries_6m": inq,
            "age": age,
            "default": d,
        })

    with open("data/applicants.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    dr = sum(r["default"] for r in rows) / N
    print(f"generated {N} applicants, default rate {dr:.1%}")


if __name__ == "__main__":
    main()
