"""Hedge baseline for the 50,000-bushel sale.

At the start of each eligible week we compare four positions:
  - no futures,
  - a short position of 10 contracts (the 1:1 hedge),
  - a static minimum-variance short position, with the hedge ratio estimated once on 2010-2019,
  - a rolling minimum-variance short position, re-estimated each week on the most recent 104
    eligible weeks completed by the hedge date.

The hedge ratio is the slope from regressing weekly cash price changes on same-contract futures price
changes. The slope multiplied by 10 gives the contract count, which we round to the nearest integer
with a floor of zero.

Sale proceeds equal 50,000 times the sale-date cash bid plus 5,000 times the number of short contracts
times the decline in the futures price. Subtracting the cash value of the crop at hedge entry and
dividing by 50,000 gives the weekly change in effective sale price per bushel:
    change = cash change - (contracts / 10) x futures change
Our main measure is the percentage reduction in the variance of this change relative to no hedge.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import CONTRACT_BU, FULL_HEDGE, N_BOOT, PROCESSED, ROLL_WINDOW, SALE_BU, SEED, TABLES


def contracts(h):
    """Contracts for a hedge ratio h: 10 x h rounded to the nearest whole contract, never below zero."""
    return max(0, int(np.floor(FULL_HEDGE * h + 0.5)))


def ols_slope(ds, df):
    fit = sm.OLS(ds, sm.add_constant(df)).fit()
    return fit.params[1], fit.bse[1], fit.rsquared, int(fit.nobs)


def load_pairs():
    p = pd.read_csv(PROCESSED / "eligibility_log.csv", parse_dates=["entry_date", "sale_date", "fnd"])
    p = p[p["eligible"]].reset_index(drop=True)
    p["dS"] = 100 * (p["cash_sale"] - p["cash_entry"])      # cash change, cents per bushel
    p["dF"] = 100 * (p["fut_sale"] - p["fut_entry"])        # futures change, cents per bushel
    return p


def rolling_ratios(p, rows):
    """Rolling hedge ratio for each hedge week, using only weeks whose sale date is on or before the hedge date."""
    h, first = [], []
    for i in rows:
        done = p[p["sale_date"] <= p.at[i, "entry_date"]].tail(ROLL_WINDOW)
        assert len(done) == ROLL_WINDOW
        h.append(ols_slope(done["dS"].to_numpy(), done["dF"].to_numpy())[0])
        first.append(done["entry_date"].iloc[0])
    return np.array(h), first


def performance(ds, dv, rng):
    """Mean, standard deviation, variance reduction (with a bootstrap 95% interval) and worst week for each position."""
    idx = rng.integers(0, len(ds), size=(N_BOOT, len(ds)))
    var0 = ds[idx].var(axis=1, ddof=1)
    rows = []
    for name, x in dv.items():
        vr = 1 - x.var(ddof=1) / ds.var(ddof=1)
        boot = 1 - x[idx].var(axis=1, ddof=1) / var0
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"strategy": name, "weeks": len(x), "mean_dV_c_bu": x.mean(), "sd_dV_c_bu": x.std(ddof=1),
                     "variance_reduction_pct": 100 * vr, "vr_ci_low_pct": 100 * lo, "vr_ci_high_pct": 100 * hi,
                     "worst_week_c_bu": x.min(), "sd_weekly_proceeds_usd": x.std(ddof=1) * SALE_BU / 100})
    return pd.DataFrame(rows), idx


def main(verbose=True):
    p = load_pairs()
    k = CONTRACT_BU / SALE_BU          # each contract covers one tenth of the sale
    tr, te = p[p["period"] == "train"], p[p["period"] == "test"]

    # static hedge ratio from the training years
    h_s, se_s, r2_s, n_s = ols_slope(tr["dS"].to_numpy(), tr["dF"].to_numpy())
    N_s = contracts(h_s)
    # rolling hedge ratio for every test week
    test_rows = te.index.to_numpy()
    h_r, win_start = rolling_ratios(p, test_rows)
    N_r = np.array([contracts(h) for h in h_r])

    ds_te, df_te = te["dS"].to_numpy(), te["dF"].to_numpy()
    dv_test = {
        "No hedge (0 contracts)": ds_te,
        f"1:1 hedge ({FULL_HEDGE} contracts)": ds_te - k * FULL_HEDGE * df_te,
        f"Static min-variance ({N_s} contracts)": ds_te - k * N_s * df_te,
        f"Rolling min-variance ({N_r.min()}-{N_r.max()} contracts)": ds_te - k * N_r * df_te,
    }
    ds_tr, df_tr = tr["dS"].to_numpy(), tr["dF"].to_numpy()
    dv_train = {
        "No hedge (0 contracts)": ds_tr,
        f"1:1 hedge ({FULL_HEDGE} contracts)": ds_tr - k * FULL_HEDGE * df_tr,
        f"Static min-variance ({N_s} contracts, in sample)": ds_tr - k * N_s * df_tr,
    }
    rng = np.random.default_rng(SEED)
    perf_te, idx = performance(ds_te, dv_test, rng)
    perf_tr, _ = performance(ds_tr, dv_train, rng)
    perf = pd.concat([perf_tr.assign(sample="train 2010-2019"), perf_te.assign(sample="test 2022-2025")])
    perf = perf[["sample"] + [c for c in perf.columns if c != "sample"]]

    # does the minimum-variance hedge beat the 1:1 hedge in the test years? (same bootstrap draws)
    names = list(dv_test)
    var0 = ds_te[idx].var(axis=1, ddof=1)
    vr_b = {n: 1 - dv_test[n][idx].var(axis=1, ddof=1) / var0 for n in names}
    diffs = {n: 100 * (vr_b[n] - vr_b[names[1]]) for n in names[2:]
             if not (n.startswith("Static") and N_s == FULL_HEDGE)}

    est = pd.DataFrame([{
        "sample": "train 2010-2019", "pairs": n_s, "h_star": h_s, "se_h": se_s, "r2": r2_s,
        "N_star_unrounded": FULL_HEDGE * h_s, "N_star": N_s,
        "rolling_h_min_test": h_r.min(), "rolling_h_median_test": np.median(h_r), "rolling_h_max_test": h_r.max(),
        "rolling_N_min_test": N_r.min(), "rolling_N_max_test": N_r.max()}])

    # week-by-week ledger for the test years, so the proceeds can be checked by hand or in Excel
    ledger = te[["entry_date", "sale_date", "contract", "cash_entry", "cash_sale", "fut_entry", "fut_sale", "dS", "dF"]].copy()
    ledger["rolling_window_start"] = win_start
    ledger["h_rolling"] = h_r
    ledger["N_rolling"] = N_r
    for lab, N in [("none", 0), ("1to1", FULL_HEDGE), ("static", N_s)]:
        ledger[f"proceeds_{lab}_usd"] = SALE_BU * ledger["cash_sale"] + CONTRACT_BU * N * (ledger["fut_entry"] - ledger["fut_sale"])
    ledger["proceeds_rolling_usd"] = SALE_BU * ledger["cash_sale"] + CONTRACT_BU * N_r * (ledger["fut_entry"] - ledger["fut_sale"])

    est.to_csv(TABLES / "table2_hedge_ratio.csv", index=False)
    perf.to_csv(TABLES / "table3_hedge_performance.csv", index=False)
    ledger.to_csv(PROCESSED / "hedge_test_weekly_ledger.csv", index=False, date_format="%Y-%m-%d")

    if verbose:
        print("\nTable 2: minimum-variance hedge ratio, weekly cash change regressed on futures change (cents/bu), 2010-2019")
        print(f"  h* = {h_s:.4f} (SE {se_s:.4f}), R2 = {r2_s:.4f}, {n_s} weeks")
        print(f"  contracts = 10 x {h_s:.4f} = {FULL_HEDGE * h_s:.2f}, rounded to {N_s}"
              + (" (the same position as the 1:1 hedge)" if N_s == FULL_HEDGE else ""))
        print(f"  rolling {ROLL_WINDOW}-week hedge ratio in the test years: lowest {h_r.min():.4f}, median {np.median(h_r):.4f}, "
              f"highest {h_r.max():.4f}; contracts {N_r.min()} to {N_r.max()} "
              f"(weeks at each count: {pd.Series(N_r).value_counts().sort_index().to_dict()})")
        print("  standard deviation of the weekly basis change (what the 1:1 hedge leaves): "
              f"train {np.std(ds_tr - df_tr, ddof=1):.2f}, test {np.std(ds_te - df_te, ddof=1):.2f} cents/bu")
        print("\nTable 3: weekly change in effective sale price (cents/bu); VR = variance reduction relative to no hedge")
        print(f"  {'sample':<16s}{'position':<44s}{'weeks':>6s}{'mean':>8s}{'SD':>8s}{'VR %':>8s}{'95% CI':>17s}{'worst':>9s}{'SD $':>9s}")
        for _, r in perf.iterrows():
            print(f"  {r['sample']:<16s}{r['strategy']:<44s}{r['weeks']:>6d}{r['mean_dV_c_bu']:>8.2f}{r['sd_dV_c_bu']:>8.2f}"
                  f"{r['variance_reduction_pct']:>8.2f}   [{r['vr_ci_low_pct']:5.1f}, {r['vr_ci_high_pct']:5.1f}]"
                  f"{r['worst_week_c_bu']:>9.2f}{r['sd_weekly_proceeds_usd']:>9.0f}")
        for n, d in diffs.items():
            lo, hi = np.percentile(d, [2.5, 97.5])
            print(f"  test years: VR of {n.split(' (')[0].lower()} minus VR of 1:1 = "
                  f"{100 * (1 - dv_test[n].var(ddof=1) / ds_te.var(ddof=1)) - perf_te['variance_reduction_pct'].iloc[1]:.2f} "
                  f"percentage points, bootstrap 95% CI [{lo:.2f}, {hi:.2f}]")
    return p, est, perf


if __name__ == "__main__":
    main()
