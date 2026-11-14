"""Command line: python -m talk.cli <stage> [options]   (run from talk_score/)

Stages, in order:
    universe-github  build the firm-year list from the public ticker history (fallback)
    firms            match the firm list to SEC CIKs
    download         fetch 10-Ks and extract Items 1, 1A, 7
    score            LM and/or FinBERT sentence scoring
    panel            firm-year Talk panel (+ percentiles if industry codes are given)
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

FIRST_YEAR, LAST_YEAR = 2010, 2024


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="talk", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data", type=Path)
    p.add_argument("--first-year", type=int, default=FIRST_YEAR)
    p.add_argument("--last-year", type=int, default=LAST_YEAR)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="stage", required=True)

    g = sub.add_parser("universe-github", help="firm-years from the public fja05680/sp500 history")
    g.add_argument("history_csv", nargs="?", help="local copy of the history CSV (default: download)")
    g.add_argument("--no-wikipedia", action="store_true",
                   help="do not add names/CIKs from Wikipedia's S&P 500 page")
    g.add_argument("--out", default=None, help="default: <data-dir>/universe.csv")

    uni_args = argparse.ArgumentParser(add_help=False)
    uni_args.add_argument("--universe", required=True, help="group firm list (csv/xlsx)")
    uni_args.add_argument("--id-col")
    uni_args.add_argument("--name-col")
    uni_args.add_argument("--ticker-col")
    uni_args.add_argument("--year-col")

    f = sub.add_parser("firms", parents=[uni_args], help="match firms to SEC CIKs")
    f.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT"))
    f.add_argument("--overrides", help="csv with firm_id,cik (default: <data-dir>/firm_cik_overrides.csv)")
    f.add_argument("--no-history-names", action="store_true",
                   help="skip EDGAR's full name list (faster, misses delisted firms)")

    d = sub.add_parser("download", help="download 10-Ks and extract sections")
    d.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT"))
    d.add_argument("--limit-firms", type=int, help="only the first N CIKs (testing)")

    s = sub.add_parser("score", help="score sentences")
    s.add_argument("--method", choices=["lm", "finbert", "both"], default="both")
    s.add_argument("--lm-dict", help="Loughran-McDonald Master Dictionary CSV")
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--no-prefilter", action="store_true",
                   help="FinBERT on every sentence (use with --sample to check prefilter recall)")
    s.add_argument("--sample", type=int, help="score only N filings")

    pn = sub.add_parser("panel", parents=[uni_args], help="build the firm-year panel")
    pn.add_argument("--industry-col", help="TRBC column in the universe file (e.g. trbc)")
    pn.add_argument("--industry-file", help="separate csv with firm_id[,year] and the TRBC column")
    pn.add_argument("--industry-digits", type=int, default=2)
    pn.add_argument("--industry-ff12", action="store_true",
                    help="no TRBC yet: use Fama-French 12 industries from SEC SIC codes")
    pn.add_argument("--out", default=None, help="default: <data-dir>/talk_panel.csv")

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(message)s")
    a.data_dir.mkdir(parents=True, exist_ok=True)
    {"universe-github": _universe_github, "firms": _firms, "download": _download,
     "score": _score, "panel": _panel}[a.stage](a)


def _client(a):
    from .edgar import EdgarClient

    if not a.user_agent:
        raise SystemExit('Set --user-agent "Your Name your@email" (or SEC_USER_AGENT)')
    return EdgarClient(a.user_agent)


def _load_universe(a) -> pd.DataFrame:
    from .firms import load_universe

    return load_universe(a.universe, a.id_col, a.name_col, a.ticker_col, a.year_col)


def _universe_github(a):
    import io

    import requests

    from .firms import GITHUB_HISTORY_URL, WIKIPEDIA_URL, universe_from_github

    headers = {"User-Agent": "BMF5324 student research (talk score)"}
    history = a.history_csv
    if history is None:
        r = requests.get(GITHUB_HISTORY_URL, headers=headers, timeout=60)
        r.raise_for_status()
        history = io.StringIO(r.text)
    html = None
    if not a.no_wikipedia:
        try:
            r = requests.get(WIKIPEDIA_URL, headers=headers, timeout=60)
            r.raise_for_status()
            html = r.text
            uni = universe_from_github(history, a.first_year, a.last_year, html)
        except Exception as e:  # noqa: BLE001 - tickers only is still usable
            print(f"WARNING: Wikipedia names not added ({e!r}); more firms will need manual review.")
            html = None
            if hasattr(history, "seek"):
                history.seek(0)
    if html is None:
        uni = universe_from_github(history, a.first_year, a.last_year)
    out = a.out or a.data_dir / "universe.csv"
    uni.to_csv(out, index=False)
    firms = uni.drop_duplicates("firm_id")
    print(f"{len(firms)} tickers, {len(uni)} firm-years -> {out}")
    if html:
        print(f"  with name: {firms['name'].notna().sum()}, with CIK: {firms['cik'].notna().sum()}, "
              f"ticker renames found: {firms['renamed_to'].notna().sum()}")


def _firms(a):
    from .firms import match_firms

    client = _client(a)
    uni = _load_universe(a)
    over_path = Path(a.overrides) if a.overrides else a.data_dir / "firm_cik_overrides.csv"
    overrides = pd.read_csv(over_path, dtype={"firm_id": str}) if over_path.exists() else None
    lookup = None if a.no_history_names else client.cik_lookup()
    matches, review = match_firms(uni, client.company_tickers(), lookup, overrides)
    matches.to_csv(a.data_dir / "firm_cik_map.csv", index=False)
    if review.empty:
        review = pd.DataFrame(columns=["firm_id", "name", "ticker"])
    review.to_csv(a.data_dir / "firms_review.csv", index=False)
    n = uni["firm_id"].nunique()
    print(f"matched {len(matches)}/{n} firms -> firm_cik_map.csv")
    print(matches["match_method"].value_counts().to_string() if len(matches) else "")
    if len(review):
        print(f"{len(review)} firms need a manual check -> firms_review.csv\n"
              f"Add firm_id,cik rows to {over_path} and rerun `firms`.")


def _download(a):
    from .pipeline import download

    cik_map = pd.read_csv(a.data_dir / "firm_cik_map.csv")
    if a.limit_firms:
        keep = sorted(cik_map["cik"].dropna().astype(int).unique())[: a.limit_firms]
        cik_map = cik_map[cik_map["cik"].isin(keep)]
    download(_client(a), cik_map, a.data_dir, a.first_year, a.last_year)


def _score(a):
    from . import pipeline

    if a.method in ("lm", "both"):
        from .lm import LMDictionary

        if not a.lm_dict:
            raise SystemExit("--lm-dict path/to/Loughran-McDonald_MasterDictionary.csv is required for LM")
        pipeline.score_lm(a.data_dir, LMDictionary.from_csv(a.lm_dict))
    if a.method in ("finbert", "both"):
        from .finbert import FinBertScorer

        pipeline.score_finbert(a.data_dir, FinBertScorer(batch_size=a.batch_size),
                               prefilter=not a.no_prefilter, sample=a.sample)


def _panel(a):
    from .panel import add_percentiles, build_panel

    uni = _load_universe(a)
    cik_map = pd.read_csv(a.data_dir / "firm_cik_map.csv", dtype={"firm_id": str})
    panel = build_panel(a.data_dir, uni, cik_map)

    industry = None
    if a.industry_file:
        ind = pd.read_csv(a.industry_file, dtype=str)
        col = a.industry_col or next(c for c in ind.columns if c.lower().startswith("trbc"))
        ind = ind.rename(columns={col: "_industry"})
        keys = ["firm_id", "year"] if "year" in ind.columns else ["firm_id"]
        if "year" in keys:
            ind["year"] = pd.to_numeric(ind["year"]).astype("Int64")
        ind = ind.drop_duplicates(keys, keep="last")
        industry = panel[keys].merge(ind[keys + ["_industry"]], on=keys, how="left")["_industry"]
    elif a.industry_col:
        industry = panel[a.industry_col]
    elif a.industry_ff12:
        from .panel import ff12

        info = pd.read_csv(a.data_dir / "firm_info.csv").drop_duplicates("cik").set_index("cik")
        industry = panel["cik"].map(info["sic"]).map(ff12)
    if industry is not None:
        panel = add_percentiles(panel, industry, a.industry_digits)
    else:
        print("No industry codes given: raw Talk only (add --industry-col, --industry-file or --industry-ff12).")

    out = a.out or a.data_dir / "talk_panel.csv"
    panel.to_csv(out, index=False)
    print(f"{len(panel)} firm-years, Talk available for {panel['fb_talk_raw'].notna().sum() if 'fb_talk_raw' in panel else 0}"
          f" (FinBERT) -> {out}")


if __name__ == "__main__":
    main()
