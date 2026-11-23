"""Current market snapshot for the market context section of our proposal.

From our own data pulls we report:
  - the latest Illinois cash bid, futures price and basis, against the average November basis in
    2010-2025 and the low point of this year's harvest season;
  - who holds CBOT corn futures in the latest CFTC Commitments of Traders report.
"""
import pandas as pd

from config import COT_CSV, RAW_CSV, TABLES

GROUPS = [("Producer/merchant/processor/user", "prod_merc_positions_long", "prod_merc_positions_short"),
          ("Swap dealers", "swap_positions_long_all", "swap__positions_short_all"),
          ("Managed money", "m_money_positions_long_all", "m_money_positions_short_all"),
          ("Other reportables", "other_rept_positions_long", "other_rept_positions_short"),
          ("Non-reportable (small traders)", "nonrept_positions_long_all", "nonrept_positions_short_all")]


def main(verbose=True):
    raw = pd.read_csv(RAW_CSV, parse_dates=["date"]).sort_values("date")
    last = raw.iloc[-1]
    hist = raw[(raw["date"].dt.year.between(2010, 2025)) & (raw["date"].dt.month == 11)]
    season = raw[(raw["date"] >= f"{last['date'].year}-09-01")]
    low = season.loc[season["basis"].idxmin()]

    cot = pd.read_csv(COT_CSV, parse_dates=["report_date_as_yyyy_mm_dd"]).sort_values("report_date_as_yyyy_mm_dd")
    c = cot.iloc[-1]
    oi = c["open_interest_all"]
    rows = [{"item": "Illinois cash bid ($/bu)", "value": last["bid"], "date": last["date"]},
            {"item": f"CBOT {last['contract_month']} futures ($/bu)", "value": last["futures_price"], "date": last["date"]},
            {"item": "Illinois basis ($/bu)", "value": last["basis"], "date": last["date"]},
            {"item": "Average November basis 2010-2025 ($/bu)", "value": hist["basis"].mean(), "date": pd.NaT},
            {"item": "Lowest basis since September 1 ($/bu)", "value": low["basis"], "date": low["date"]},
            {"item": "COT open interest (contracts)", "value": oi, "date": c["report_date_as_yyyy_mm_dd"]}]
    for name, lo, sh in GROUPS:
        net = c[lo] - c[sh]
        rows.append({"item": f"COT net position, {name} (contracts)", "value": net, "date": c["report_date_as_yyyy_mm_dd"]})
        rows.append({"item": f"COT net position, {name} (% of open interest)", "value": 100 * net / oi,
                     "date": c["report_date_as_yyyy_mm_dd"]})
    snap = pd.DataFrame(rows)
    snap.to_csv(TABLES / "table6_market_snapshot.csv", index=False, date_format="%Y-%m-%d")

    if verbose:
        print("\nTable 6: market snapshot")
        print(f"  Illinois, week ending {last['date']:%Y-%m-%d}: cash bid {last['bid']:.2f}, "
              f"{last['contract_month']} futures {last['futures_price']:.4f}, basis {last['basis']:+.4f} $/bu")
        print(f"  average November basis 2010-2025: {hist['basis'].mean():+.4f} $/bu ({len(hist)} weeks)")
        print(f"  lowest basis since September 1: {low['basis']:+.4f} $/bu, week ending {low['date']:%Y-%m-%d}")
        print(f"  CFTC COT, CBOT corn, positions on {c['report_date_as_yyyy_mm_dd']:%Y-%m-%d}: open interest {oi:,.0f} contracts")
        for name, lo, sh in GROUPS:
            net = c[lo] - c[sh]
            print(f"    {name:<34s} long {c[lo]:>9,.0f}  short {c[sh]:>9,.0f}  net {net:>+10,.0f} ({100 * net / oi:+.1f}% of OI)")
    return snap


if __name__ == "__main__":
    main()
