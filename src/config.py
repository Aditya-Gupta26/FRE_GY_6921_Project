"""Settings shared by all baseline scripts.

Our case is an Illinois corn grower who sells 50,000 bushels to an elevator one week after placing
a short hedge in CBOT corn futures. We also forecast next week's Illinois elevator bid.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "usda_ams_grain_basis_illinois_corn.csv"
RAW_INFO = ROOT / "data" / "raw" / "retrieval_info.json"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

# USDA AMS Grain Basis dataset on Open Ag Transport Data
DATASET_PAGE = "https://agtransport.usda.gov/Grain/Grain-Basis/v85y-3hep"
API_CSV = "https://agtransport.usda.gov/resource/v85y-3hep.csv"
API_META = "https://agtransport.usda.gov/api/views/v85y-3hep.json"
MARKET, MARKET_TYPE, COMMODITY = "Illinois", "Elevator Bid", "Corn"

# CFTC Commitments of Traders, disaggregated futures-only report, CBOT corn (contract code 002602)
COT_PAGE = "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm"
COT_API = "https://publicreporting.cftc.gov/resource/72hh-3qpy.csv"
COT_CODE = "002602"
COT_CSV = ROOT / "data" / "raw" / "cftc_cot_corn_disaggregated.csv"

# We study January 8, 2010 to December 26, 2025, the same window as our initial proposal.
START, END = "2010-01-08", "2025-12-26"
# Train on 2010-2019, select settings on 2020-2021, keep 2022-2025 for the out-of-sample test.
TRAIN_LAST_YEAR = 2019
VALID_LAST_YEAR = 2021

SALE_BU = 50_000                      # bushels the grower sells
CONTRACT_BU = 5_000                   # bushels in one CBOT corn contract
FULL_HEDGE = SALE_BU // CONTRACT_BU   # 10 contracts match the sale volume
ROLL_WINDOW = 104                     # most recent eligible weeks used for the rolling hedge ratio

# CBOT corn delivery months
CORN_MONTHS = {"Mar": 3, "May": 5, "Jul": 7, "Sep": 9, "Dec": 12}

N_BOOT = 2000                         # bootstrap draws for confidence intervals
SEED = 6921                           # fixed so the bootstrap gives the same numbers on every run


def period(year):
    """Which part of the sample a year belongs to."""
    if year <= TRAIN_LAST_YEAR:
        return "train"
    if year <= VALID_LAST_YEAR:
        return "validation"
    return "test"
