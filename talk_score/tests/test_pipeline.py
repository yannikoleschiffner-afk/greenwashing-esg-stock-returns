import pandas as pd
import pytest

from fixtures import GREEN, NEUTRAL_ENV, FakeClassifier, FakeEdgar, make_10k, make_10k_mda_elsewhere, make_exhibit13
from talk import pipeline
from talk.finbert import FinBertScorer
from talk.lm import LMDictionary
from talk.panel import add_percentiles, build_panel, industry_code, percentile_within


def _filing(acc, filed, period, doc="doc.htm"):
    return {"accession": acc, "filing_date": filed, "period_of_report": period,
            "form": "10-K", "primary_document": doc}


@pytest.fixture
def data(tmp_path):
    """Three firms:
      cik 1: green talker, Dec FYE, FY2015 + FY2016 (+ FY2008 outside sample)
      cik 2: January FYE retailer, one neutral env sentence
      cik 3: document download fails
      cik 4: MD&A in Exhibit 13
    """
    filings = {
        1: [_filing("1-15", "2016-02-20", "2015-12-31"), _filing("1-16", "2017-02-20", "2016-12-31"),
            _filing("1-08", "2009-02-20", "2008-12-31")],
        2: [_filing("2-15", "2016-03-20", "2016-01-30")],
        3: [_filing("3-15", "2016-02-20", "2015-12-31")],
        4: [_filing("4-15", "2016-02-20", "2015-12-31"), _filing("4-xx", "2017-02-20", float("nan"))],
    }
    docs = {(1, "1-15"): make_10k(GREEN), (1, "1-16"): make_10k(GREEN[:1]),
            (2, "2-15"): make_10k(NEUTRAL_ENV),
            (4, "4-15"): make_10k_mda_elsewhere([], exhibit13=True),
            (4, "4-15", "EX-13"): make_exhibit13(GREEN)}
    cik_map = pd.DataFrame({"firm_id": ["F1", "F2", "F3", "F1b", "F4"], "cik": [1, 2, 3, 1, 4],
                            "sec_name": ["A", "B", "C", "A", "D"], "match_method": "name"})
    pipeline.download(FakeEdgar(filings, docs), cik_map, tmp_path, 2010, 2024)

    lm_path = tmp_path / "lm.csv"
    pd.DataFrame({"Word": ["ACHIEVED", "IMPROVE", "STRENGTHEN"], "Negative": 0,
                  "Positive": 2009}).to_csv(lm_path, index=False)
    pipeline.score_lm(tmp_path, LMDictionary.from_csv(lm_path))
    pipeline.score_finbert(tmp_path, FinBertScorer(esg=FakeClassifier.esg, tone=FakeClassifier.tone))
    return tmp_path, cik_map


def test_download_stage(data):
    d, _ = data
    meta = pipeline.load_filing_meta(d).set_index("accession")
    assert sorted(meta.index) == ["1-15", "1-16", "2-15", "4-15"]  # FY2008 skipped, 3-15 failed
    assert meta.loc["4-15", "item7_source"] == "ex13"
    assert meta.loc["1-15", "item7_source"] == "item7"
    assert meta.loc["2-15", "year"] == 2015  # January FYE -> previous year
    assert meta.loc["1-15", "section_source"] == "items"
    errors = pd.read_csv(d / "download_errors.csv")
    assert errors["accession"].tolist() == ["3-15", "4-xx"]  # 4-xx: no period of report
    assert pd.read_csv(d / "firm_info.csv")["cik"].tolist() == [1, 2, 3, 4]


def test_scores_and_resume(data):
    d, _ = data
    fb = pd.read_csv(d / "scores_finbert.csv").set_index("accession")
    assert fb.loc["1-15", "fb_n_env_pos"] == 2
    assert fb.loc["1-16", "fb_n_env_pos"] == 1
    assert fb.loc["2-15", "fb_n_env"] == 1 and fb.loc["2-15", "fb_n_env_pos"] == 0
    lm = pd.read_csv(d / "scores_lm.csv").set_index("accession")
    assert lm.loc["1-15", "lm_n_env_pos"] == 2
    # rerunning does not duplicate rows
    pipeline.score_finbert(d, FinBertScorer(esg=FakeClassifier.esg, tone=FakeClassifier.tone))
    assert fb.loc["4-15", "fb_n_env_pos"] == 2  # found via Exhibit 13
    assert len(pd.read_csv(d / "scores_finbert.csv")) == 4


def test_panel(data):
    d, cik_map = data
    uni = pd.DataFrame({
        "firm_id": ["F1", "F1", "F2", "F3", "F1b"],
        "name": ["A", "A", "B", "C", "A class B"],
        "ticker": None,
        "year": pd.array([2015, 2016, 2015, 2015, 2015], dtype="Int64"),
        "trbc": ["5010101010", "5010101010", "5010202020", "5010101010", "5010101010"],
    })
    panel = build_panel(d, uni, cik_map)
    assert len(panel) == 5  # every universe firm-year kept
    p = panel.set_index(["firm_id", "year"])
    assert p.loc[("F1", 2015), "return_year"] == 2016
    assert p.loc[("F3", 2015), "fb_talk_raw"] != p.loc[("F3", 2015), "fb_talk_raw"]  # NaN, not 0
    r = p.loc[("F1", 2015)]
    assert r["fb_talk_raw"] == pytest.approx(r["fb_env_share"] * r["fb_env_pos_share"])
    assert r["fb_talk_raw"] == pytest.approx(2 / r["n_sentences"])
    assert p.loc[("F1b", 2015), "secondary_share_class"]
    assert not p.loc[("F1", 2015), "secondary_share_class"]

    panel = add_percentiles(panel, panel["trbc"])
    p = panel.set_index(["firm_id", "year"])
    # 2015, sector "50": F1 (talk) and F2 (zero positive talk) are ranked; F3 missing.
    assert p.loc[("F1", 2015), "talk_pct"] == 100
    assert p.loc[("F2", 2015), "talk_pct"] == 0
    assert p.loc[("F1b", 2015), "talk_pct"] == 100  # inherits company value
    assert pd.isna(p.loc[("F3", 2015), "talk_pct"])
    assert p.loc[("F1", 2015), "industry_year_n"] == 2
    assert p.loc[("F1", 2016), "talk_pct"] == 50  # alone in its group


def test_percentile_within_ties_and_groups():
    df = pd.DataFrame({"g": ["a"] * 5 + ["b"] * 2, "v": [1, 2, 2, 3, None, 5, 7]})
    pct = percentile_within(df, "v", ["g"]).tolist()
    assert pct[:4] == [0, 50, 50, 100]
    assert pd.isna(pct[4])
    assert pct[5:] == [0, 100]


def test_industry_code():
    s = pd.Series(["5010101010", 5010101010.0, "Energy", None])
    assert industry_code(s).tolist()[:3] == ["50", "50", "Energy"]


def test_cli_panel(data, tmp_path):
    from talk.cli import main

    d, _ = data
    uni_path = tmp_path / "universe.csv"
    pd.DataFrame({
        "ISIN": ["F1", "F1", "F2", "F3"], "Name": ["A", "A", "B", "C"], "Ticker": ["@A", "@A", "B", "C"],
        "Year": [2015, 2016, 2015, 2015], "TRBC": ["5010101010", "5010101010", "5120202020", "5010101010"],
    }).to_csv(uni_path, index=False)
    pd.DataFrame({"firm_id": ["F1", "F2", "F3"], "cik": [1, 2, 3], "sec_name": "x",
                  "match_method": "name"}).to_csv(d / "firm_cik_map.csv", index=False)
    main(["--data-dir", str(d), "panel", "--universe", str(uni_path), "--industry-col", "TRBC"])
    out = pd.read_csv(d / "talk_panel.csv")
    assert len(out) == 4
    assert {"talk_pct", "talk_pct_lm", "fb_talk_raw", "lm_talk_raw", "return_year"} <= set(out.columns)
    assert out.loc[(out.firm_id == "F2") & (out.year == 2015), "talk_pct"].item() == 50  # own sector


def test_ff12():
    from talk.panel import ff12

    assert [ff12(s) for s in [2911, 4911, 6021, 7372, 3674, 2834, 5331, 1311, "", None, 9999]] == \
        ["Enrgy", "Utils", "Money", "BusEq", "BusEq", "Hlth", "Shops", "Enrgy", None, None, "Other"]
