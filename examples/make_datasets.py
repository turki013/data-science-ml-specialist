"""Generate the bundled EXAMPLE datasets used by tests/test_example_*.py.

Why synthetic: every file here has a pathology planted on purpose (a post-outcome column, a
re-listed near-duplicate, a broken-free A/B assignment, a weekly seasonality). Real data cannot be
shipped in a public repo, and a generator makes the pathology explicit and reviewable.

Rules for these files:
* Deterministic. Fixed seeds, no clock, no network. Re-running this script reproduces the CSVs
  byte-for-byte on the same library versions (that is also why the tests read the committed CSVs
  instead of regenerating at test time).
* Small. A few hundred KB total; they are fixtures, not a benchmark.
* Documented. `references/datasets.md` says what each file contains, what is wrong with it, which
  test proves the corresponding guard fires, and what a senior would notice.
* Clearly fake. No real people, no scraped listings, no customer data.

    python examples/make_datasets.py            # writes examples/data/*.csv
    python examples/make_datasets.py --check    # writes to a temp dir and diffs against the committed files
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

for _stream in (sys.stdout, sys.stderr):     # the listings file contains Arabic
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

CHURN_SEED = 11
SALES_SEED = 12
LISTINGS_SEED = 13
AB_SEED = 14
RANKING_SEED = 15


# ------------------------------------------------------------------------------- churn (classification)
def churn(seed: int = CHURN_SEED) -> pd.DataFrame:
    """Subscription churn with TWO planted leaks and repeated customers.

    Columns
    -------
    customer_id            repeated per monthly snapshot -> the split must be grouped
    tenure_months          honest feature
    monthly_spend          honest feature, missing for ~8% of rows (MAR on older snapshots)
    plan, region           honest categoricals
    support_calls_90d      honest feature
    support_calls_after_cancel   LEAK: recorded only after the cancellation event
    cancelled_at           LEAK: timestamp that exists exactly for churned = 1
    churned                target
    """
    rng = np.random.default_rng(seed)
    n_customers = 900
    plans = np.array(["basic", "pro", "enterprise"])
    plan_p = np.array([0.5, 0.35, 0.15])
    regions = np.array(["riyadh", "jeddah", "dammam", "abha"])

    rows = []
    plan_of = rng.choice(plans, size=n_customers, p=plan_p)
    region_of = rng.choice(regions, size=n_customers)
    base_spend = rng.gamma(shape=4.0, scale=45.0, size=n_customers) + 40.0
    base_tenure = rng.integers(1, 48, size=n_customers)
    call_rate = rng.gamma(shape=2.0, scale=0.6, size=n_customers)

    for c in range(n_customers):
        snapshots = int(rng.integers(1, 4))          # 1-3 monthly snapshots per customer
        for s in range(snapshots):
            tenure = int(base_tenure[c] + s)
            spend = float(base_spend[c] * (1 + 0.01 * s))
            calls = int(rng.poisson(call_rate[c]))

            logit = (-2.4
                     + 0.05 * calls
                     - 0.055 * tenure
                     - 0.010 * (spend - 180) / 10
                     + (0.9 if plan_of[c] == "basic" else 0.0)
                     + (0.35 if region_of[c] == "abha" else 0.0))
            p_churn = 1 / (1 + np.exp(-logit))
            churned = int(rng.random() < p_churn)

            # LEAK 1: recorded after the outcome, so it is a proxy for the label itself.
            after_calls = int(rng.poisson(4.5)) if churned else 0
            # LEAK 2: a timestamp that only exists for churners.
            cancelled_at = (pd.Timestamp("2025-06-01") + pd.Timedelta(days=int(rng.integers(0, 365)))
                            if churned else pd.NaT)

            spend_out = np.nan if (s == 0 and rng.random() < 0.08) else round(spend, 2)
            rows.append((f"C{c:04d}", tenure, spend_out, plan_of[c], region_of[c], calls,
                         after_calls, cancelled_at, churned))

    df = pd.DataFrame(rows, columns=[
        "customer_id", "tenure_months", "monthly_spend", "plan", "region", "support_calls_90d",
        "support_calls_after_cancel", "cancelled_at", "churned"])
    df["cancelled_at"] = df["cancelled_at"].dt.strftime("%Y-%m-%d").replace("NaT", "")
    return df


# ------------------------------------------------------------------------------- sales (forecasting)
def sales(seed: int = SALES_SEED) -> pd.DataFrame:
    """Two years of daily sales with weekly seasonality, a mild trend, promo spikes and noise.

    The point is the SPLIT and the BASELINE: a random split of this frame leaks the future, and a
    plain naive forecast loses badly to a seasonal-naive one.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    n = len(dates)
    weekday = dates.dayofweek.to_numpy()
    weekly = np.array([1.00, 0.95, 1.02, 1.05, 1.25, 1.30, 0.90])   # Fri/Sat peak
    trend = np.linspace(0.0, 0.18, n)
    promo = (rng.random(n) < 0.05)
    promo_boost = np.where(promo, rng.uniform(1.15, 1.45, n), 1.0)
    noise = rng.normal(1.0, 0.05, n)
    y = 900 * weekly[weekday] * (1 + trend) * promo_boost * noise
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"),
                         "sales": np.round(y, 1),
                         "promo": promo.astype(int)})


# ------------------------------------------------------------------------------- listings (Arabic text)
def listings(seed: int = LISTINGS_SEED) -> pd.DataFrame:
    """Arabic real-estate listings where ~30% of properties were RE-LISTED with a slightly edited title.

    The pathology: a random split puts the same property on both sides. `dstk.leakage.text_group_ids`
    recovers the duplicate groups from the normalized title.
    """
    rng = np.random.default_rng(seed)
    districts = ["حي النرجس", "حي الملقا", "حي العليا", "حي الشفا", "حي الياسمين"]
    kinds = ["فيلا", "شقة", "شقة", "أرض"]
    ar_digits = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

    rows = []
    listing_id = 1000
    used_titles: set[tuple[str, str, int, int]] = set()   # keeps one property per normalized title
    for prop in range(150):
        district = str(rng.choice(districts))
        kind = str(rng.choice(kinds))
        rooms = int(rng.integers(2, 7))
        size = int(rng.integers(80, 500))
        # Two different properties with the same (kind, rooms, district, size) would produce the SAME
        # title and therefore the same duplicate group: bump the area until the title is unique, so a
        # group means "this property was re-listed", which is what the test asserts.
        while (kind, district, rooms, size) in used_titles:
            size += 1
        used_titles.add((kind, district, rooms, size))
        price = float(np.round(size * rng.uniform(2600, 5200) + rooms * 9000, -3))
        day = pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(rng.integers(0, 240)))

        relists = int(rng.choice([1, 2, 3], p=[0.7, 0.2, 0.1]))
        root = f"{kind} {rooms} غرف في {district}"
        for r in range(relists):
            # Same property, different surface text: Arabic digits, punctuation, spacing, and a small
            # price edit after the first listing.
            if r == 0:
                title = f"{root}، مساحة {size} م2"
            elif r == 1:
                title = f"{root} - مساحة {str(size).translate(ar_digits)} م٢"
            else:
                title = f"  {root}!!  مساحة {size} م2  "
            rows.append((f"L{listing_id}", title, district, kind, rooms, size,
                         price if r == 0 else float(round(price * (1 + rng.uniform(-0.03, 0.03)), -3)),
                         (day + pd.Timedelta(days=int(rng.integers(0, 30)))).strftime("%Y-%m-%d"),
                         r))
            listing_id += 1

    df = pd.DataFrame(rows, columns=["listing_id", "title", "district", "kind", "rooms",
                                     "size_m2", "price_sar", "listed_on", "relist_index"])
    return df.sort_values(["listed_on", "listing_id"], kind="stable").reset_index(drop=True)


# ------------------------------------------------------------------------------- A/B experiment
def ab(seed: int = AB_SEED) -> pd.DataFrame:
    """A clean 50/50 experiment with a SIMULATED treatment effect on conversion, a correlated pre-period covariate
    (for CUPED), and a guardrail metric that moves the wrong way (support tickets).

    Pathology: the guardrail, not the primary metric, is the interesting part.
    """
    rng = np.random.default_rng(seed)
    n_per_arm = 2000
    groups = np.array(["control"] * n_per_arm + ["treatment"] * n_per_arm)
    is_treat = groups == "treatment"

    pre_spend = np.round(rng.gamma(shape=2.5, scale=55.0, size=groups.size), 2)
    z = (pre_spend - pre_spend.mean()) / pre_spend.std()

    p_base = 1 / (1 + np.exp(-(-2.55 + 0.55 * z)))                  # conversion depends on pre_spend
    p = np.where(is_treat, p_base * 1.30, p_base)                   # simulated +2pp treatment effect
    converted = (rng.random(groups.size) < p).astype(int)

    # A continuous engagement metric that is strongly correlated with the pre-period covariate:
    # this is where CUPED earns its keep (variance reduction ~0.75).
    week1_minutes = np.round(20 + 0.08 * pre_spend + np.where(is_treat, 1.2, 0.0)
                             + rng.normal(0, 4.0, groups.size), 1)
    tickets = rng.poisson(np.where(is_treat, 0.42, 0.35))           # guardrail: moves the wrong way
    days_to_convert = np.where(converted == 1, rng.integers(1, 15, groups.size), 0)

    return pd.DataFrame({"user_id": [f"U{i:05d}" for i in range(groups.size)],
                         "group": groups, "converted": converted,
                         "pre_spend": pre_spend, "week1_minutes": week1_minutes,
                         "support_tickets": tickets, "days_to_convert": days_to_convert})


# ------------------------------------------------------------------------------- ranking interactions
def interactions(seed: int = RANKING_SEED) -> pd.DataFrame:
    """Implicit feedback (user -> item, timestamped) with a cluster structure: users consume 1-2 of
    8 topics, items belong to a topic. Item-item CF should therefore beat the popularity baseline,
    which is exactly the comparison a ranking review should make first.
    """
    rng = np.random.default_rng(seed)
    n_topics, items_per_topic, n_users = 8, 12, 260
    items = np.arange(n_topics * items_per_topic)
    topic_of = items // items_per_topic

    rows = []
    base = pd.Timestamp("2025-03-01")
    for u in range(n_users):
        chosen = rng.choice(n_topics, size=int(rng.integers(1, 3)), replace=False)
        pool = items[np.isin(topic_of, chosen)]
        if rng.random() < 0.4:   # a bit of noise: some cross-topic clicks
            pool = np.union1d(pool, rng.choice(items, size=4, replace=False))
        n_ev = int(rng.integers(4, 26))
        picks = rng.choice(pool, size=min(n_ev, pool.size), replace=False)
        times = base + pd.to_timedelta(np.sort(rng.integers(0, 120 * 24, size=picks.size)), unit="h")
        for item, ts in zip(picks, times):
            rows.append((f"U{u:03d}", f"I{int(item):02d}", ts.strftime("%Y-%m-%d %H:%M")))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "ts"])


# ------------------------------------------------------------------------------------------------- main
BUILDERS = {"churn_leaky.csv": churn, "sales_daily.csv": sales, "listings_ar.csv": listings,
            "experiment_ab.csv": ab, "interactions.csv": interactions}


def write_all(out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, fn in BUILDERS.items():
        path = out / name
        fn().to_csv(path, index=False, lineterminator="\n")
        written.append(path)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--check", action="store_true",
                    help="regenerate into a temp dir and fail if any committed file differs")
    a = ap.parse_args()

    if a.check:
        with tempfile.TemporaryDirectory() as tmp:
            fresh = write_all(Path(tmp))
            changed = []
            for p in fresh:
                target = DATA / p.name
                if not target.exists() or target.read_bytes().replace(b"\r\n", b"\n") != p.read_bytes().replace(b"\r\n", b"\n"):
                    changed.append(p.name)
            if changed:
                print("datasets drifted from the committed files:", ", ".join(changed))
                print("regenerate with `python examples/make_datasets.py` and review the diff.")
                return 1
            print(f"OK: {len(fresh)} example datasets reproduce byte-for-byte")
            return 0

    for p in write_all(Path(a.out)):
        with p.open(encoding="utf-8") as f:
            rows = sum(1 for _ in f) - 1
        print(f"{p.name:22s} {p.stat().st_size:>8,} bytes  {rows:>5,} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
