"""Clean the Illinois corn series and decide which weeks can enter the hedge backtest.

Steps:
1. Keep January 8, 2010 to December 26, 2025 and check that the basis equals cash bid minus futures.
2. Infer the delivery year of each futures quote from the quote date and contract month, and check it
   against the CBOT corn calendar (CBOT Rulebook, Chapter 10). Labels that fail are excluded.
3. Flag isolated one-week spikes in the basis as suspected recording errors.
4. Form pairs of weeks (hedge placed in the first week, corn sold in the second) and keep a pair only
   if the two weeks are exactly seven days apart, use the same futures contract, pass the checks
   above, and the sale happens before first notice day.

Calendar rules we use (weekends are skipped, exchange holidays are not):
  first notice day = last business day of the month before the delivery month
  last trading day = business day before the 15th calendar day of the delivery month
"""
import pandas as pd

from config import CORN_MONTHS, END, PROCESSED, RAW_CSV, START, TABLES, period

# A week is a suspected recording error when the basis jumps by at least 40 cents and comes back
# to within 5 cents of its old level the next week.
SPIKE_C, BACK_C = 40, 5


def prev_business_day(d):
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def first_notice_day(year, month):
    return prev_business_day(pd.Timestamp(year, month, 1) - pd.Timedelta(days=1))


def last_trading_day(year, month):
    return prev_business_day(pd.Timestamp(year, month, 15) - pd.Timedelta(days=1))


def listed_contracts(date):
    """CBOT corn contracts still trading on a date, nearest first, as (year, month)."""
    out = []
    for y in (date.year, date.year + 1):
        for m in sorted(CORN_MONTHS.values()):
            if last_trading_day(y, m) >= date:
                out.append((y, m))
    return out


def load_weekly():
    raw = pd.read_csv(RAW_CSV, parse_dates=["date"])
    w = raw[(raw["date"] >= START) & (raw["date"] <= END)]
    w = w[["date", "bid", "futures_price", "basis", "contract_month"]].drop_duplicates("date")
    w = w.sort_values("date").reset_index(drop=True)
    w = w.rename(columns={"bid": "cash", "futures_price": "futures"})

    rows = []
    for r in w.itertuples():
        d, lab = r.date, r.contract_month
        info = {"contract": "", "fnd": pd.NaT, "ltd": pd.NaT, "label_ok": False, "label_note": ""}
        if lab not in CORN_MONTHS:
            info["label_note"] = "nonstandard contract label"
        else:
            # a March label quoted in November refers to next year's March contract
            m = CORN_MONTHS[lab]
            y = d.year if m >= d.month else d.year + 1
            info.update(contract=f"{lab}-{y}", fnd=first_notice_day(y, m), ltd=last_trading_day(y, m))
            live = listed_contracts(d)
            if (y, m) not in live:
                info["label_note"] = "contract not trading on quote date"
            elif live.index((y, m)) > 1:
                info["label_note"] = "not one of the two nearest contracts"
            else:
                info["label_ok"] = True
        rows.append(info)
    w = pd.concat([w, pd.DataFrame(rows)], axis=1)
    w["in_delivery_period"] = w["label_ok"] & (w["date"] >= w["fnd"])
    w["suspect"] = suspected_errors(w)
    w["period"] = w["date"].dt.year.map(period)
    return w


def suspected_errors(w):
    """Flag a week when the basis jumps and then returns, while the futures contract stays the same.

    The futures price does not share the jump, so the cash bid for that week is most likely mistyped.
    We exclude such weeks rather than guess the true bid.
    """
    b = 100 * w["basis"]
    weekly = (w["date"].diff().dt.days == 7) & (w["date"].diff(-1).dt.days == -7)
    one_contract = (w["contract"] == w["contract"].shift()) & (w["contract"] == w["contract"].shift(-1))
    jump = (b - b.shift()).abs() >= SPIKE_C
    back = (b.shift(-1) - b.shift()).abs() <= BACK_C
    return weekly & one_contract & jump & back


def build_pairs(w):
    """One row per pair of consecutive weeks: the hedge is placed in the first, the corn is sold in the second."""
    a, b = w.iloc[:-1].reset_index(drop=True), w.iloc[1:].reset_index(drop=True)
    p = pd.DataFrame({
        "entry_date": a["date"], "sale_date": b["date"],
        "gap_days": (b["date"] - a["date"]).dt.days,
        "entry_label": a["contract_month"], "sale_label": b["contract_month"],
        "contract": a["contract"], "fnd": a["fnd"],
        "cash_entry": a["cash"], "cash_sale": b["cash"],
        "fut_entry": a["futures"], "fut_sale": b["futures"],
        "basis_entry": a["basis"], "basis_sale": b["basis"],
    })
    # the first rule a pair fails is recorded as its reason
    reason = pd.Series("", index=p.index)
    reason[p["gap_days"] != 7] = "not exactly 7 days apart"
    same = (p["entry_label"] == p["sale_label"]) & (a["contract"] == b["contract"])
    reason[(reason == "") & ~same] = "futures contract changes within the week"
    bad = ~(a["label_ok"] & b["label_ok"])
    reason[(reason == "") & bad] = "contract label fails calendar check"
    reason[(reason == "") & (a["suspect"] | b["suspect"])] = "suspected recording error"
    reason[(reason == "") & (p["sale_date"] >= p["fnd"])] = "sale on or after first notice day"
    p["eligible"] = reason == ""
    p["reason"] = reason.where(reason != "", "eligible")
    p["period"] = p["entry_date"].dt.year.map(period)
    return p


def funnel(w, p):
    steps = [
        ("Weekly observations, " + START + " to " + END, len(w)),
        ("Pairs of consecutive weeks", len(p)),
        ("Pairs exactly seven days apart", int((p["gap_days"] == 7).sum())),
        ("Same futures contract in both weeks", int(((p["gap_days"] == 7) & (p["reason"] != "futures contract changes within the week")).sum())),
        ("Contract labels pass the CBOT calendar check",
         int(p["reason"].isin(["eligible", "suspected recording error", "sale on or after first notice day"]).sum())),
        ("No suspected recording error", int(p["reason"].isin(["eligible", "sale on or after first notice day"]).sum())),
        ("Sale before first notice day (eligible)", int(p["eligible"].sum())),
    ]
    out = pd.DataFrame(steps, columns=["step", "count"])
    for per in ["train", "validation", "test"]:
        out.loc[len(out)] = [f"Eligible pairs, {per}", int((p["eligible"] & (p["period"] == per)).sum())]
    return out


def main(verbose=True):
    w = load_weekly()
    p = build_pairs(w)
    f = funnel(w, p)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    w.to_csv(PROCESSED / "weekly_clean.csv", index=False, date_format="%Y-%m-%d")
    p.to_csv(PROCESSED / "eligibility_log.csv", index=False, date_format="%Y-%m-%d")
    f.to_csv(TABLES / "table1_sample_funnel.csv", index=False)
    if verbose:
        print("Data checks")
        print(f"  {len(w)} weeks, {w['date'].nunique()} unique dates, {int(w[['cash', 'futures', 'basis']].isna().sum().sum())} missing values")
        print(f"  largest gap between basis and cash minus futures: {(w['basis'] - (w['cash'] - w['futures'])).abs().max():.1e} $/bu")
        print(f"  quote days: {w['date'].dt.day_name().value_counts().to_dict()}")
        for _, r in w[~w["label_ok"]].iterrows():
            print(f"  excluded label: {r['date']:%Y-%m-%d} '{r['contract_month']}' ({r['label_note']}), futures {r['futures']:.4f}")
        for i in w.index[w["suspect"]]:
            print(f"  suspected recording error: {w.at[i, 'date']:%Y-%m-%d} cash bid {w.at[i, 'cash']:.2f} "
                  f"(weeks before and after {w.at[i - 1, 'cash']:.2f}, {w.at[i + 1, 'cash']:.2f}); futures "
                  f"{w.at[i - 1, 'futures']:.4f}, {w.at[i, 'futures']:.4f}, {w.at[i + 1, 'futures']:.4f}; basis "
                  f"{w.at[i - 1, 'basis']:+.4f}, {w.at[i, 'basis']:+.4f}, {w.at[i + 1, 'basis']:+.4f}")
        print(f"  quotes on or after first notice day: {int(w['in_delivery_period'].sum())}")
        print("\nTable 1: sample funnel")
        for _, r in f.iterrows():
            print(f"  {r['step']:<55s} {r['count']:>4d}")
        print("  reasons for exclusion:", p.loc[~p["eligible"], "reason"].value_counts().to_dict())
    return w, p


if __name__ == "__main__":
    main()
