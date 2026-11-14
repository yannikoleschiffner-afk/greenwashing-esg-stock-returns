"""Firm universe and firm -> SEC CIK matching.

Universe (all firms in the S&P 500 at any year-end 2010-2024, including firms
that later left the index):
  * Preferred: the group's Datastream list (one row per firm and year-end,
    e.g. historical constituent lists such as LS&PCOMP1215 for Dec 2015). Using the
    same list as the return data guarantees the datasets merge.
  * Fallback: `universe_from_github` builds the same firm-year list from the
    public fja05680/sp500 ticker history. It has tickers only (no names), so
    more firms end up in the manual review file.

Sample rule: firm-year (i, t) is in the sample if firm i was in the S&P 500 on
31 December of year t, the last day before the return year t+1 starts.

Matching to CIK (needed to download 10-Ks), in this order:
  1. manual overrides file (firm_id, cik)
  2. ticker in SEC's current ticker list, if the names also agree
  3. fuzzy name match against SEC's current list, then against every EDGAR
     entity name ever used (covers delisted / acquired / renamed firms)
Anything uncertain goes to a review file with the best candidates.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

ACCEPT_NAME_SCORE = 90
TICKER_NAME_SCORE = 60

# Datastream abbreviates names ("INTL.BUS.MCHS.", "UNITEDHEALTH GP.").
_ABBREVIATIONS = {
    "intl": "international", "int": "international", "bus": "business", "mchs": "machines",
    "gp": "group", "grp": "group", "gen": "general", "amer": "american", "am": "american",
    "techs": "technologies", "tech": "technology", "pharms": "pharmaceuticals",
    "pharm": "pharmaceutical", "finl": "financial", "fin": "financial", "svs": "services",
    "svcs": "services", "ser": "services", "inds": "industries", "ind": "industries",
    "cos": "companies", "res": "resources", "prds": "products", "prod": "products",
    "sys": "systems", "mng": "mining", "entm": "entertainment", "ent": "entertainment",
    "comms": "communications", "comm": "communications", "com": "communications",
    "hldg": "holding", "hldgs": "holdings", "mgmt": "management", "mgt": "management",
    "natl": "national", "nat": "national", "elec": "electric", "eng": "engineering",
    "ppty": "property", "pptys": "properties", "tst": "trust", "tr": "trust",
    "bncp": "bancorp", "bcp": "bancorp", "ins": "insurance", "labs": "laboratories",
    "lab": "laboratories", "mnfg": "manufacturing", "mfg": "manufacturing",
    "utils": "utilities", "util": "utilities", "chm": "chemical", "chem": "chemical",
    "chems": "chemicals", "dev": "development", "equip": "equipment", "hlth": "health",
    "hltcr": "healthcare", "invs": "investments", "inv": "investment", "mkts": "markets",
    "pwr": "power", "sci": "scientific", "stl": "steel", "trans": "transportation",
    "wrld": "world", "wwide": "worldwide", "brands": "brands", "sts": "states",
}
_STOPWORDS = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "plc",
    "llc", "lp", "the", "sa", "nv", "ag", "se", "class", "cl", "ord", "shs", "new", "old",
    "del", "de", "reit", "and", "adr", "registered", "common", "stock", "fully", "paid",
    "holdings", "holding", "group", "cos", "companies",
}


def normalise_name(name: str) -> str:
    s = str(name).lower()
    s = re.sub(r"/[a-z]{2,3}/?", " ", s)  # EDGAR state tags: "APPLE INC /CA/"
    s = re.sub(r"\(.*?\)", " ", s)
    s = s.replace("&", " and ").replace("'", "")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = [_ABBREVIATIONS.get(t, t) for t in s.split()]
    toks = [t for t in toks if t not in _STOPWORDS]
    # trailing share class letters: "ALPHABET A", "BERKSHIRE HATHAWAY B"
    while len(toks) > 1 and len(toks[-1]) == 1:
        toks.pop()
    return " ".join(toks)


def normalise_ticker(ticker: str) -> str | None:
    """Datastream / RIC / Bloomberg / plain tickers -> SEC style (BRK-B)."""
    if ticker is None or pd.isna(ticker):
        return None
    t = str(ticker).strip()
    if not t:
        return None
    t = re.sub(r"^[A-Z]{1,2}:", "", t)  # Datastream "U:IBM"
    t = t.lstrip("@")  # Datastream "@AAPL" (Nasdaq)
    t = re.sub(r"\s+(US|UN|UW|UQ|UA)(\s+EQUITY)?$", "", t, flags=re.I)  # Bloomberg
    m = re.match(r"^([A-Za-z]+?)([a-z]?)\.(O|N|OQ|K|A|P|Z|PK)$", t)  # RIC "BRKb.N"
    if m:
        t = m.group(1) + (f"-{m.group(2)}" if m.group(2) else "")
    t = t.upper().replace(".", "-").replace("/", "-").replace(" ", "-")
    return t or None


# --- universe ------------------------------------------------------------------

def load_universe(
    path: str | Path,
    id_col: str | None = None,
    name_col: str | None = None,
    ticker_col: str | None = None,
    year_col: str | None = None,
) -> pd.DataFrame:
    """Read the group's firm list; returns columns firm_id, name, ticker, year (+ extras).

    Column names are guessed if not given (case-insensitive): id -> isin / dscd /
    firm_id / ric, name -> name / company, ticker -> ticker / symbol, year -> year.
    """
    df = pd.read_csv(path, dtype=str) if str(path).endswith(".csv") else pd.read_excel(path, dtype=str)
    lower = {c.lower(): c for c in df.columns}

    def pick(given, options):
        if given:
            return given
        return next((lower[o] for o in options if o in lower), None)

    id_col = pick(id_col, ["firm_id", "isin", "dscd", "datastream_code", "ric", "cusip", "ticker"])
    name_col = pick(name_col, ["name", "company", "company_name", "firm_name"])
    ticker_col = pick(ticker_col, ["ticker", "symbol", "tic", "wc05601"])
    year_col = pick(year_col, ["year", "fyear", "fiscal_year"])
    if id_col is None:
        raise ValueError(f"No firm id column found in {path}; pass --id-col")

    out = pd.DataFrame({
        "firm_id": df[id_col].str.strip(),
        "name": df[name_col] if name_col else None,
        "ticker": df[ticker_col] if ticker_col else None,
        "year": pd.to_numeric(df[year_col], errors="coerce").astype("Int64") if year_col else pd.NA,
    })
    extras = [c for c in df.columns if c not in {id_col, name_col, ticker_col, year_col}]
    for c in extras:
        out[c] = df[c]
    return out.dropna(subset=["firm_id"])


GITHUB_HISTORY_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def universe_from_github(
    history_csv: str | Path,
    first_year: int,
    last_year: int,
    wikipedia_html: str | None = None,
) -> pd.DataFrame:
    """Firm-years from fja05680/sp500 'S&P 500 Historical Components & Changes'.

    For each year, the last listed date on or before 31 December gives the
    members. firm_id is the ticker as of that date.

    The history has tickers only. With `wikipedia_html` (the Wikipedia "List of
    S&P 500 companies" page) we add:
      * name and CIK of current members (Wikipedia lists the CIK),
      * name of firms that were removed (the page's table of index changes),
      * ticker renames (FB -> META): a ticker that leaves the list on the same
        day another one joins, with no index change recorded on Wikipedia, is
        the same company; it gets the new ticker's name and CIK.
    """
    hist = pd.read_csv(history_csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    members = [set(t.strip() for t in s.split(",")) for s in hist["tickers"]]
    rows = []
    for year in range(first_year, last_year + 1):
        idx = hist.index[hist["date"] <= f"{year}-12-31"]
        if len(idx) == 0:
            continue
        for t in sorted(members[idx[-1]]):
            rows.append({"firm_id": t, "name": None, "ticker": t, "year": year, "cik": pd.NA})
    out = pd.DataFrame(rows)
    out["year"] = out["year"].astype("Int64")
    if wikipedia_html:
        out = _add_wikipedia_info(out, hist, members, wikipedia_html)
    return out


def parse_wikipedia_tables(html: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(current members: ticker, name, cik; changes: date, added/removed ticker+name)."""
    from io import StringIO

    tables = pd.read_html(StringIO(html))
    current = changes = None
    for t in tables:
        cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
        low = [c.lower() for c in cols]
        if current is None and any("symbol" in c for c in low) and any("cik" in c for c in low):
            t.columns = cols
            current = pd.DataFrame({
                "ticker": t[cols[next(i for i, c in enumerate(low) if "symbol" in c)]],
                "name": t[cols[next(i for i, c in enumerate(low) if "security" in c)]],
                "cik": pd.to_numeric(t[cols[next(i for i, c in enumerate(low) if "cik" in c)]],
                                     errors="coerce").astype("Int64"),
            })
        elif changes is None and any(c.startswith("removed") for c in low) and any(c.startswith("added") for c in low):
            def col(prefix, kind):
                return t.iloc[:, next(i for i, c in enumerate(low) if c.startswith(prefix) and kind in c)]
            date_col = t.iloc[:, next(i for i, c in enumerate(low) if "date" in c)]
            changes = pd.DataFrame({
                "date": pd.to_datetime(date_col, errors="coerce", format="mixed"),
                "added_ticker": col("added", "ticker"), "added_name": col("added", "security"),
                "removed_ticker": col("removed", "ticker"), "removed_name": col("removed", "security"),
            })
    if current is None or changes is None:
        raise ValueError("Wikipedia page layout not recognised")
    for df in (current, changes):
        for c in df.columns:
            if c.endswith("ticker"):
                df[c] = df[c].map(normalise_ticker)
    return current, changes


def _add_wikipedia_info(out, hist, members, html) -> pd.DataFrame:
    current, changes = parse_wikipedia_tables(html)
    cur = current.drop_duplicates("ticker").set_index("ticker")
    final_members = {normalise_ticker(t) for t in members[-1]}

    # first/last date each ticker is listed
    first_seen, last_seen = {}, {}
    for i, tickers in enumerate(members):
        for t in tickers:
            first_seen.setdefault(t, i)
            last_seen[t] = i

    def recorded(kind: str, ticker: str, when: pd.Timestamp) -> pd.Series | None:
        """Index change on Wikipedia for this ticker within ~4 weeks of `when`."""
        c = changes[(changes[f"{kind}_ticker"] == normalise_ticker(ticker))
                    & ((changes["date"] - when).abs() <= pd.Timedelta(days=28))]
        return None if c.empty else c.iloc[0]

    info: dict[str, dict] = {}
    # Current members: name + CIK straight from Wikipedia.
    for t in members[-1]:
        n = normalise_ticker(t)
        if n in cur.index:
            info[t] = {"name": cur.at[n, "name"], "cik": cur.at[n, "cik"]}

    # Leavers, oldest exit last so rename chains (A -> B -> C) resolve.
    leavers = sorted((t for t in last_seen if normalise_ticker(t) not in final_members),
                     key=lambda t: -last_seen[t])
    for t in leavers:
        i = last_seen[t]
        exit_date = hist["date"].iloc[i + 1] if i + 1 < len(hist) else hist["date"].iloc[i]
        removal = recorded("removed", t, exit_date)
        if removal is not None:
            info[t] = {"name": removal["removed_name"], "cik": pd.NA}
            continue
        # Rename: exactly one ticker enters on the day this one leaves and
        # Wikipedia records no index addition for it.
        new = [u for u in members[i + 1] if first_seen[u] == i + 1] if i + 1 < len(members) else []
        new = [u for u in new if recorded("added", u, exit_date) is None]
        if len(new) == 1 and new[0] in info:
            info[t] = {**info[new[0]], "renamed_to": new[0]}
            continue
        # Otherwise: the name from when it was added, if recorded.
        entry = hist["date"].iloc[first_seen[t]]
        added = recorded("added", t, entry)
        if added is not None:
            info[t] = {"name": added["added_name"], "cik": pd.NA}

    out = out.copy()
    out["name"] = out["firm_id"].map(lambda t: info.get(t, {}).get("name"))
    out["cik"] = out["firm_id"].map(lambda t: info.get(t, {}).get("cik", pd.NA)).astype("Int64")
    out["renamed_to"] = out["firm_id"].map(lambda t: info.get(t, {}).get("renamed_to"))
    return out


# --- CIK matching ----------------------------------------------------------------

def match_firms(
    universe: pd.DataFrame,
    sec_tickers: pd.DataFrame,
    cik_lookup: pd.DataFrame | None = None,
    overrides: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (matches, review). One row per firm_id in each."""
    universe = universe.copy()
    for c in ("cik", "renamed_to"):
        if c not in universe:
            universe[c] = pd.NA
    universe["cik"] = pd.to_numeric(universe["cik"], errors="coerce")
    firms = (
        universe.sort_values("year", na_position="first")
        .groupby("firm_id", as_index=False)
        .agg(name=("name", "last"), ticker=("ticker", "last"), given_cik=("cik", "last"),
             renamed_to=("renamed_to", "last"),
             first_year=("year", "min"), last_year=("year", "max"))
    )
    sec_tickers = sec_tickers.assign(
        ticker_norm=sec_tickers["ticker"].map(normalise_ticker),
        name_norm=sec_tickers["sec_name"].map(normalise_name),
    )
    by_ticker = sec_tickers.drop_duplicates("ticker_norm").set_index("ticker_norm")
    current = sec_tickers.drop_duplicates("cik")[["cik", "sec_name", "name_norm"]]
    all_names = current
    if cik_lookup is not None and len(cik_lookup):
        hist = cik_lookup.assign(name_norm=cik_lookup["sec_name"].map(normalise_name))
        all_names = pd.concat([current, hist], ignore_index=True).drop_duplicates(["cik", "name_norm"])
    blocks = _block_index(all_names)
    latest_year = firms["last_year"].max()
    over = {}
    if overrides is not None and len(overrides):
        over = dict(zip(overrides["firm_id"].astype(str), overrides["cik"].astype(int)))

    matches, review = [], []
    for f in firms.itertuples(index=False):
        name_norm = normalise_name(f.name) if isinstance(f.name, str) and f.name else ""
        tick = normalise_ticker(f.ticker)
        base = {"firm_id": f.firm_id, "name": f.name, "ticker": f.ticker,
                "first_year": f.first_year, "last_year": f.last_year}

        if f.firm_id in over:
            matches.append({**base, "cik": over[f.firm_id], "sec_name": None,
                            "match_method": "override", "match_score": 100.0})
            continue

        # 1b. CIK supplied with the firm list (Wikipedia, for current members
        # and for tickers renamed into a current member)
        if pd.notna(f.given_cik):
            renamed = isinstance(f.renamed_to, str) and f.renamed_to
            matches.append({**base, "cik": int(f.given_cik), "sec_name": None,
                            "match_method": f"renamed_to_{f.renamed_to}" if renamed else "given_cik",
                            "match_score": float("nan")})
            continue

        # 2. ticker, confirmed by name when we have one
        ticker_cand = None
        if tick and tick in by_ticker.index:
            cand = by_ticker.loc[tick]
            ticker_cand = (int(cand["cik"]), cand["sec_name"], float("nan"))
            if name_norm:
                score = fuzz.token_set_ratio(name_norm, cand["name_norm"])
                if score >= TICKER_NAME_SCORE:
                    matches.append({**base, "cik": int(cand["cik"]), "sec_name": cand["sec_name"],
                                    "match_method": "ticker", "match_score": score})
                    continue
            elif pd.notna(f.last_year) and f.last_year >= latest_year:
                # No name to confirm, but the firm is still in the index at the
                # end of the sample, so today's ticker is its own. Tickers of
                # firms that left may since have been reused (Sprint's "S").
                matches.append({**base, "cik": int(cand["cik"]), "sec_name": cand["sec_name"],
                                "match_method": "ticker_only", "match_score": float("nan")})
                continue

        # 3. name
        cands = _name_candidates(name_norm, blocks, all_names) if name_norm else []
        if cands and cands[0][2] >= ACCEPT_NAME_SCORE and (len(cands) == 1 or cands[0][2] > cands[1][2]):
            cik, sec_name, score = cands[0]
            matches.append({**base, "cik": cik, "sec_name": sec_name,
                            "match_method": "name", "match_score": score})
            continue

        if ticker_cand and all(c[0] != ticker_cand[0] for c in cands):
            cands = [ticker_cand] + cands
        row = {**base}
        for i, (cik, sec_name, score) in enumerate(cands[:3], 1):
            row.update({f"cand{i}_cik": cik, f"cand{i}_name": sec_name, f"cand{i}_score": score})
        review.append(row)

    return pd.DataFrame(matches), pd.DataFrame(review)


def _block_index(names: pd.DataFrame) -> dict[str, list[int]]:
    blocks: dict[str, list[int]] = {}
    for i, n in enumerate(names["name_norm"]):
        if n:
            blocks.setdefault(n.split()[0], []).append(i)
    return blocks


def _name_candidates(name_norm: str, blocks, names: pd.DataFrame) -> list[tuple[int, str, float]]:
    idx = blocks.get(name_norm.split()[0], [])
    if not idx:
        return []
    sub = names.iloc[idx]
    hits = process.extract(name_norm, sub["name_norm"].tolist(), scorer=fuzz.token_sort_ratio, limit=10)
    seen, out = set(), []
    for _, score, j in hits:
        row = sub.iloc[j]
        cik = int(row["cik"])
        if cik in seen:
            continue
        seen.add(cik)
        out.append((cik, row["sec_name"], float(score)))
    return out
