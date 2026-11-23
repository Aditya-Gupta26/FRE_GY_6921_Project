"""Download the Illinois corn elevator bids from the USDA AMS Grain Basis dataset, and the latest
CFTC Commitments of Traders positions in CBOT corn.

Both sources have public APIs, so no key or login is needed. We save the full Illinois corn series
(2007 to the latest week), the last 52 weekly COT reports for corn, and a small file that records
when we pulled them.

Run: python src/download_data.py
"""
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from config import (API_CSV, API_META, COMMODITY, COT_API, COT_CODE, COT_CSV, COT_PAGE, DATASET_PAGE, MARKET,
                    MARKET_TYPE, RAW_CSV, RAW_INFO)

COT_COLUMNS = ["report_date_as_yyyy_mm_dd", "market_and_exchange_names", "open_interest_all",
               "prod_merc_positions_long", "prod_merc_positions_short",
               "swap_positions_long_all", "swap__positions_short_all",
               "m_money_positions_long_all", "m_money_positions_short_all",
               "other_rept_positions_long", "other_rept_positions_short",
               "nonrept_positions_long_all", "nonrept_positions_short_all"]


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main():
    # keep only the Illinois elevator bid for corn, oldest week first
    query = {
        "$where": f"market_name='{MARKET}' AND market_type='{MARKET_TYPE}' AND commodity='{COMMODITY}'",
        "$order": "date",
        "$limit": "50000",
    }
    url = API_CSV + "?" + urllib.parse.urlencode(query)
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    RAW_CSV.write_bytes(fetch(url))

    # last 52 weekly COT reports for CBOT corn
    cot_query = {
        "$select": ",".join(COT_COLUMNS),
        "$where": f"cftc_contract_market_code='{COT_CODE}'",
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": "52",
    }
    cot_url = COT_API + "?" + urllib.parse.urlencode(cot_query)
    COT_CSV.write_bytes(fetch(cot_url))

    # record the pull date and when USDA last updated the dataset
    meta = json.loads(fetch(API_META))
    df = pd.read_csv(RAW_CSV, parse_dates=["date"])
    cot = pd.read_csv(COT_CSV, parse_dates=["report_date_as_yyyy_mm_dd"])
    info = {
        "dataset": "USDA AMS Grain Basis (" + DATASET_PAGE + ")",
        "filter": query["$where"],
        "request_url": url,
        "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "dataset_last_updated_utc": datetime.fromtimestamp(meta["rowsUpdatedAt"], timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "rows": len(df),
        "first_date": df["date"].min().strftime("%Y-%m-%d"),
        "last_date": df["date"].max().strftime("%Y-%m-%d"),
        "cot_dataset": "CFTC Commitments of Traders, disaggregated futures only (" + COT_PAGE + ")",
        "cot_request_url": cot_url,
        "cot_reports": len(cot),
        "cot_latest_report_date": cot["report_date_as_yyyy_mm_dd"].max().strftime("%Y-%m-%d"),
    }
    RAW_INFO.write_text(json.dumps(info, indent=2) + "\n")
    print(f"Saved {len(df)} weeks ({info['first_date']} to {info['last_date']}) to {RAW_CSV.name}")
    print(f"Saved {len(cot)} COT reports (latest {info['cot_latest_report_date']}) to {COT_CSV.name}")


if __name__ == "__main__":
    main()
