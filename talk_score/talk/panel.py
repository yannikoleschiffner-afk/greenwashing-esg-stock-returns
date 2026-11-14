"""Firm-year Talk panel and industry-year percentiles."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import load_filing_meta

LATE_FILING_DAYS = 90


def filing_scores(data_dir: Path) -> pd.DataFrame:
    """One row per 10-K with metadata and raw Talk measures."""
    data_dir = Path(data_dir)
    meta = load_filing_meta(data_dir)
    if meta.empty:
        raise ValueError(f"No downloaded filings in {data_dir}")
    for name in ["scores_finbert.csv", "scores_lm.csv"]:
        p = data_dir / name
        if p.exists():
            s = pd.read_csv(p).drop_duplicates("accession", keep="last")
            s = s.drop(columns=[c for c in s.columns if c in meta.columns and c != "accession"])
            meta = meta.merge(s, on="accession", how="left")

    n = meta["n_sentences"] if "n_sentences" in meta else pd.Series(np.nan, index=meta.index)
    for pre in ["fb", "lm"]:
        if f"{pre}_n_env" not in meta:
            continue
        env, pos = meta[f"{pre}_n_env"], meta[f"{pre}_n_env_pos"]
        # talk_raw = env_share * env_pos_share
        meta[f"{pre}_talk_raw"] = _div(pos, n)
        meta[f"{pre}_env_share"] = _div(env, n)
        meta[f"{pre}_env_pos_share"] = _div(pos, env)

    meta["filing_date"] = pd.to_datetime(meta["filing_date"])
    meta["period_of_report"] = pd.to_datetime(meta["period_of_report"])
    meta["late_filer"] = (meta["filing_date"] - meta["period_of_report"]).dt.days > LATE_FILING_DAYS
    return meta


def _div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.astype(float).where(b > 0) / b.where(b > 0)


def one_filing_per_year(filings: pd.DataFrame) -> pd.DataFrame:
    """If a firm has two 10-Ks mapped to one year (e.g. after changing its
    fiscal year end), keep the later fiscal period, then the first filing."""
    filings = filings.sort_values(["cik", "year", "period_of_report", "filing_date"],
                                  ascending=[True, True, False, True])
    counts = filings.groupby(["cik", "year"])["accession"].transform("size")
    filings = filings.assign(n_10k_in_year=counts)
    return filings.drop_duplicates(["cik", "year"], keep="first")


def build_panel(data_dir: Path, universe: pd.DataFrame, cik_map: pd.DataFrame) -> pd.DataFrame:
    """Talk panel on the group's firm-year universe (keys: firm_id, year).

    Every firm-year in the universe gets a row; Talk is missing (never zero)
    when no usable 10-K was found.
    """
    filings = one_filing_per_year(filing_scores(data_dir))
    ids = cik_map[["firm_id", "cik", "sec_name", "match_method"]].copy()
    ids["firm_id"] = ids["firm_id"].astype(str)

    uni = universe.drop(columns=[c for c in ("cik", "sec_name", "match_method") if c in universe])
    uni["firm_id"] = uni["firm_id"].astype(str)
    if uni["year"].isna().all():
        # Firm-level list without years: keep every year we have a 10-K for.
        uni = uni.drop(columns="year").drop_duplicates("firm_id")
        panel = uni.merge(ids, on="firm_id", how="left").merge(filings, on="cik", how="left")
    else:
        uni = uni.drop_duplicates(["firm_id", "year"])
        panel = uni.merge(ids, on="firm_id", how="left").merge(filings, on=["cik", "year"], how="left")
    panel["return_year"] = panel["year"] + 1

    # Share classes of one company (GOOGL/GOOG, FOXA/FOX) share a CIK and so a
    # 10-K. Keep one per company-year in the regression.
    panel = panel.sort_values(["year", "cik", "firm_id"])
    panel["secondary_share_class"] = panel["cik"].notna() & panel.duplicated(["cik", "year"])

    first = ["firm_id", "name", "ticker", "cik", "sec_name", "year", "return_year",
             "accession", "filing_date", "period_of_report", "fb_talk_raw", "lm_talk_raw"]
    first = [c for c in first if c in panel.columns]
    return panel[first + [c for c in panel.columns if c not in first]].reset_index(drop=True)


def industry_code(trbc: pd.Series, digits: int = 2) -> pd.Series:
    """First `digits` digits of a numeric TRBC code (2 = economic sector).
    Non-numeric values (e.g. sector names) are used as they are."""
    s = trbc.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    numeric = s.str.fullmatch(r"\d+").fillna(False)
    return s.where(~numeric, s.str[:digits])


# Fama-French 12 industries from 4-digit SIC codes. Used as a stand-in for
# TRBC sectors until the TRBC codes are available (and as a robustness check).
_FF12 = [
    ("NoDur", [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)]),
    ("Durbl", [(2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716),
               (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)]),
    ("Manuf", [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629),
               (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799),
               (3830, 3839), (3860, 3899)]),
    ("Enrgy", [(1200, 1399), (2900, 2999)]),
    ("Chems", [(2800, 2829), (2840, 2899)]),
    ("BusEq", [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)]),
    ("Telcm", [(4800, 4899)]),
    ("Utils", [(4900, 4949)]),
    ("Shops", [(5000, 5999), (7200, 7299), (7600, 7699)]),
    ("Hlth", [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)]),
    ("Money", [(6000, 6999)]),
]


def ff12(sic) -> str | None:
    try:
        s = int(float(sic))
    except (TypeError, ValueError):
        return None
    for name, ranges in _FF12:
        if any(lo <= s <= hi for lo, hi in ranges):
            return name
    return "Other"


def percentile_within(df: pd.DataFrame, value: str, groups: list[str]) -> pd.Series:
    """0-100 percentile rank within groups; ties get the average rank.

    pct = (rank - 1) / (n - 1) * 100, so the lowest firm is 0 and the highest
    100 in every group. Single-firm groups get 50.
    """
    g = df.groupby(groups, dropna=False)[value]
    rank = g.rank(method="average")
    n = g.transform("count")
    pct = (rank - 1) / (n - 1) * 100
    return pct.where(n > 1, 50.0).where(df[value].notna())


def add_percentiles(panel: pd.DataFrame, industry: pd.Series, digits: int = 2) -> pd.DataFrame:
    """Talk percentiles within year x TRBC industry.

    Ranked among primary share classes with a Talk score; secondary share
    classes receive their company's value.
    """
    panel = panel.copy()
    panel["industry"] = industry_code(industry, digits).values
    rankable = ~panel["secondary_share_class"] & panel["industry"].notna()
    for raw, pct in [("fb_talk_raw", "talk_pct"), ("lm_talk_raw", "talk_pct_lm")]:
        if raw not in panel:
            continue
        sub = panel[rankable]
        panel[pct] = np.nan
        panel.loc[rankable, pct] = percentile_within(sub, raw, ["year", "industry"])
        company = panel[rankable & panel["cik"].notna()].set_index(["cik", "year"])[pct]
        sec = panel["secondary_share_class"]
        keys = pd.MultiIndex.from_frame(panel.loc[sec, ["cik", "year"]])
        panel.loc[sec, pct] = company.reindex(keys).values
    # Number of firms each fb percentile was ranked against.
    if "fb_talk_raw" in panel:
        counted = rankable & panel["fb_talk_raw"].notna()
        panel["industry_year_n"] = counted.groupby([panel["year"], panel["industry"]]).transform("sum")
    return panel
