import pandas as pd
import pytest

from talk.edgar import filter_ten_k, parse_cik_lookup, parse_period
from talk.firms import match_firms, normalise_name, normalise_ticker, universe_from_github


@pytest.mark.parametrize("raw,norm", [
    ("INTL.BUS.MCHS.", "international business machines"),
    ("International Business Machines Corp", "international business machines"),
    ("ALPHABET A", "alphabet"),
    ("Alphabet Inc.", "alphabet"),
    ("BERKSHIRE HATHAWAY 'B'", "berkshire hathaway"),
    ("PROCTER & GAMBLE", "procter gamble"),
    ("APPLE INC /CA/", "apple"),
    ("UNITEDHEALTH GP.", "unitedhealth"),
])
def test_normalise_name(raw, norm):
    assert normalise_name(raw) == norm


@pytest.mark.parametrize("raw,norm", [
    ("U:IBM", "IBM"), ("@AAPL", "AAPL"), ("AAPL.O", "AAPL"), ("BRKb.N", "BRK-B"),
    ("BRK.B", "BRK-B"), ("MSFT US Equity", "MSFT"), (None, None), ("", None),
])
def test_normalise_ticker(raw, norm):
    assert normalise_ticker(raw) == norm


SEC_TICKERS = pd.DataFrame({
    "cik": [320193, 51143, 1652044, 1652044, 101830, 1583708],
    "ticker": ["AAPL", "IBM", "GOOGL", "GOOG", "S", "S"],
    "sec_name": ["Apple Inc.", "INTERNATIONAL BUSINESS MACHINES CORP", "Alphabet Inc.",
                 "Alphabet Inc.", "SPRINT CORP", "SentinelOne, Inc."],
}).iloc[[0, 1, 2, 3, 5]]  # Sprint no longer listed; "S" now belongs to SentinelOne

CIK_LOOKUP = pd.DataFrame({
    "sec_name": ["SPRINT CORP", "SPRINT NEXTEL CORP", "TIME WARNER INC.", "TIME WARNER CABLE INC."],
    "cik": [101830, 101830, 1105705, 1377013],
})


def test_match_firms_ticker_name_and_delisted():
    uni = pd.DataFrame({
        "firm_id": ["US0378331005", "US4592001014", "US02079K3059", "US85207U1051", "US8873173038", "XX"],
        "name": ["APPLE", "INTL.BUS.MCHS.", "ALPHABET A", "SPRINT", "TIME WARNER", "UNKNOWN CO"],
        "ticker": ["@AAPL", "U:IBM", "@GOOGL", "U:S", "U:TWX", None],
        "year": [2024, 2024, 2024, 2019, 2017, 2012],
    })
    m, review = match_firms(uni, SEC_TICKERS, CIK_LOOKUP)
    got = dict(zip(m["firm_id"], zip(m["cik"], m["match_method"])))
    assert got["US0378331005"] == (320193, "ticker")
    assert got["US4592001014"] == (51143, "ticker")
    assert got["US02079K3059"] == (1652044, "ticker")
    # Ticker S now = SentinelOne; name check sends Sprint to the history names.
    assert got["US85207U1051"] == (101830, "name")
    assert got["US8873173038"] == (1105705, "name")
    assert list(review["firm_id"]) == ["XX"]


def test_ticker_only_match_requires_firm_still_in_index():
    uni = pd.DataFrame({"firm_id": ["AAPL", "S"], "name": [None, None],
                        "ticker": ["AAPL", "S"], "year": [2024, 2019]})
    m, review = match_firms(uni, SEC_TICKERS, CIK_LOOKUP)
    assert m.set_index("firm_id").loc["AAPL", "match_method"] == "ticker_only"
    # Sprint left in 2020; today's "S" is another firm -> manual review, with candidate
    assert list(review["firm_id"]) == ["S"]
    assert review.iloc[0]["cand1_cik"] == 1583708


def test_overrides_win():
    uni = pd.DataFrame({"firm_id": ["XX"], "name": ["UNKNOWN CO"], "ticker": [None], "year": [2012]})
    m, review = match_firms(uni, SEC_TICKERS, None, pd.DataFrame({"firm_id": ["XX"], "cik": [999]}))
    assert m.iloc[0]["cik"] == 999 and review.empty


def test_universe_from_github(tmp_path):
    p = tmp_path / "hist.csv"
    pd.DataFrame({
        "date": ["2010-06-01", "2010-12-20", "2011-03-01", "2012-01-05"],
        "tickers": ["A,B,C", "A,B,D", "A,D", "A,D,E"],
    }).to_csv(p, index=False)
    u = universe_from_github(p, 2010, 2012)
    assert sorted(u[u.year == 2010].firm_id) == ["A", "B", "D"]  # last list before 31 Dec
    assert sorted(u[u.year == 2011].firm_id) == ["A", "D"]
    assert sorted(u[u.year == 2012].firm_id) == ["A", "D", "E"]


def test_edgar_helpers():
    assert parse_period("<PERIOD>20151231\n<FILER>") == "2015-12-31"
    assert parse_period("CONFORMED PERIOD OF REPORT:\t20160130") == "2016-01-30"
    lookup = parse_cik_lookup("APPLE INC:0000320193:\nAPPLE COMPUTER INC:0000320193:\nbad line\n")
    assert lookup["cik"].tolist() == [320193, 320193]
    raw = pd.DataFrame({
        "accessionNumber": ["a", "b", "c"], "filingDate": ["2016-02-01"] * 3,
        "reportDate": ["2015-12-31", "", "2015-12-31"], "form": ["10-K", "10-K", "10-K/A"],
        "primaryDocument": ["x.htm", "y.htm", "z.htm"],
    })
    f = filter_ten_k(raw)
    assert f["accession"].tolist() == ["a", "b"]  # amendment dropped
    assert f["period_of_report"].isna().tolist() == [False, True]


def test_find_exhibit_url():
    from fixtures import EDGAR_INDEX
    from talk.edgar import find_exhibit_url

    assert find_exhibit_url(EDGAR_INDEX, "EX-13") == \
        "https://www.sec.gov/Archives/edgar/data/51143/000155837025001801/ibm-20241231xex13.htm"
    assert find_exhibit_url(EDGAR_INDEX, "EX-21") is None


WIKI_HTML = """<html><body>
<table class="wikitable" id="constituents">
<thead><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>Date added</th><th>CIK</th><th>Founded</th></tr></thead>
<tbody>
<tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td><td>1982-11-30</td><td>0000320193</td><td>1977</td></tr>
<tr><td>META</td><td>Meta Platforms</td><td>Communication Services</td><td>2013-12-23</td><td>0001326801</td><td>2004</td></tr>
<tr><td>TMUS</td><td>T-Mobile US</td><td>Communication Services</td><td>2019-07-15</td><td>0001283699</td><td>1994</td></tr>
<tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td><td>2010-02-16</td><td>0001067983</td><td>1839</td></tr>
</tbody></table>
<table class="wikitable" id="changes">
<thead>
<tr><th rowspan="2">Effective Date</th><th colspan="2">Added</th><th colspan="2">Removed</th><th rowspan="2">Reason</th></tr>
<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>
</thead>
<tbody>
<tr><td>April 6, 2020</td><td>ZZZ</td><td>Zed Corp</td><td>S</td><td>Sprint Corporation</td><td>Merger</td></tr>
<tr><td>July 15, 2019</td><td>TMUS</td><td>T-Mobile US</td><td></td><td></td><td>Market cap</td></tr>
<tr><td>March 1, 2012</td><td>OLDA</td><td>Old Added Inc.</td><td></td><td></td><td>x</td></tr>
</tbody></table></body></html>"""


def test_universe_with_wikipedia(tmp_path):
    from talk.firms import parse_wikipedia_tables

    p = tmp_path / "hist.csv"
    pd.DataFrame({
        "date": ["2011-12-30", "2012-03-01", "2013-12-30", "2019-07-15", "2020-04-06",
                 "2021-12-30", "2022-06-09", "2023-01-05", "2024-12-30"],
        "tickers": ["AAPL,FB,S,BRK.B,GONE", "AAPL,FB,S,BRK.B,OLDA", "AAPL,FB,S,BRK.B,OLDA",
                    "AAPL,FB,S,BRK.B,OLDA,TMUS", "AAPL,FB,BRK.B,OLDA,TMUS,ZZZ", "AAPL,FB,BRK.B,OLDA,TMUS,ZZZ",
                    "AAPL,META,BRK.B,OLDA,TMUS,ZZZ", "AAPL,META,BRK.B,TMUS,NEW1,NEW2",
                    "AAPL,META,BRK.B,TMUS,NEW1,NEW2"],
    }).to_csv(p, index=False)
    cur, changes = parse_wikipedia_tables(WIKI_HTML)
    assert cur.set_index("ticker").loc["BRK-B", "cik"] == 1067983
    assert changes["removed_ticker"].tolist()[0] == "S"

    u = universe_from_github(p, 2011, 2024, WIKI_HTML).drop_duplicates("firm_id").set_index("firm_id")
    assert u.loc["AAPL", "cik"] == 320193 and u.loc["BRK.B", "cik"] == 1067983
    assert u.loc["FB", "renamed_to"] == "META" and u.loc["FB", "cik"] == 1326801  # rename
    assert u.loc["S", "name"] == "Sprint Corporation" and pd.isna(u.loc["S", "cik"])  # recorded removal
    assert u.loc["OLDA", "name"] == "Old Added Inc."  # two tickers joined that day: no rename
    assert pd.isna(u.loc["OLDA", "renamed_to"])
    assert pd.isna(u.loc["GONE", "name"])  # nothing known -> manual review

    m, review = match_firms(u.reset_index(), SEC_TICKERS, CIK_LOOKUP)
    got = dict(zip(m["firm_id"], m["match_method"]))
    assert got["FB"] == "renamed_to_META" and got["AAPL"] == "given_cik"
    assert m.set_index("firm_id").loc["S", "cik"] == 101830  # Sprint via historical names
    assert "GONE" in set(review["firm_id"])
