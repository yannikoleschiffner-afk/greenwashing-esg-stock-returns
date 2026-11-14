import pandas as pd
import pytest

from fixtures import GREEN, NEUTRAL_ENV, make_10k
from talk.keywords import environmental_hits, is_environmental
from talk.lm import LMDictionary, sentence_tone, tone_counts
from talk.parse import html_to_text, parse_filing, split_sentences
from talk.years import assign_year, assign_year_series


@pytest.mark.parametrize("period,year", [
    ("2015-12-31", 2015),  # December year-end
    ("2015-09-26", 2015),  # Apple
    ("2015-06-30", 2015),  # Microsoft
    ("2015-05-31", 2015),  # Nike: NOT moved back (unlike Compustat)
    ("2016-01-30", 2015),  # Target / Home Depot
    ("2016-01-31", 2015),  # Walmart ("fiscal 2016" in its own label)
    ("2016-01-02", 2015),  # 52/53-week year ending near Dec 31
    ("2016-02-01", 2015),  # retail 52/53-week ending in first week of Feb
    ("2016-02-29", 2016),  # Constellation Brands style Feb year-end
])
def test_assign_year(period, year):
    assert assign_year(period) == year


def test_assign_year_series_matches_scalar():
    periods = pd.Series(["2015-12-31", "2016-01-30", "2016-02-07", "2016-02-08"])
    assert assign_year_series(periods).tolist() == [2015, 2015, 2015, 2016]


def test_sections_skip_toc_and_tables_and_hidden_xbrl():
    parsed = parse_filing(make_10k(GREEN, ixbrl=True))
    assert parsed.source == "items"
    s = parsed.sections
    assert "greenhouse gas emissions" in s["item1"]
    assert "RISK FACTORS" not in s["item1"]  # stops at Item 1A
    assert s["item1a"].startswith("ITEM 1A")
    assert "Investors should carefully consider" in s["item1a"]
    assert "UNRESOLVED" not in s["item1a"]
    assert s["item7"].startswith("ITEM 7.")
    assert "1,234.5" not in s["item7"]  # numeric table dropped
    assert "Interest rate risk" not in s["item7"]  # 7A excluded
    assert "HIDDEN XBRL FACT" not in parsed.text
    assert "audited statements" not in parsed.text  # Item 8 excluded


def test_fallback_to_full_text_without_headings():
    raw = "<html><body><p>" + "We sell products to customers worldwide. " * 50 + "</p></body></html>"
    parsed = parse_filing(raw)
    assert parsed.source == "fulltext"
    assert parsed.item7_short


def test_html_to_text_plain_text_passthrough():
    assert html_to_text("ITEM 1.   BUSINESS\n\n\n\nText") == "ITEM 1. BUSINESS\n\nText"


def test_split_sentences_handles_abbreviations_and_debris():
    text = ("We operate in the U.S. and Europe through Acme Inc. and its subsidiaries. "
            "Emissions fell 10% in fiscal 2015 compared with the prior year.\n"
            "ITEM 1A. RISK FACTORS\n"
            "2015 2014 2013 1,200 1,100 900 800 700\n"
            "Our results may vary.")
    sents = split_sentences(text)
    assert sents == [
        "We operate in the U.S. and Europe through Acme Inc. and its subsidiaries.",
        "Emissions fell 10% in fiscal 2015 compared with the prior year.",
    ]


@pytest.mark.parametrize("sentence,expected", [
    ("We cut greenhouse gas emissions by 20%.", True),
    ("Our solar and wind power projects expanded.", True),
    ("We are committed to protecting the environment.", True),
    ("Compliance with environmental regulations is costly.", True),
    ("The competitive environment remains challenging.", False),
    ("The economic environment in which we operate is uncertain.", False),
    ("The environment in which we operate is changing rapidly.", False),
    ("We aim for sustainable growth in earnings.", False),
    ("Our sustainability report describes our goals.", True),
    ("We produce carbon black and carbon steel.", False),
    ("Wind down of the legacy business continued.", False),
])
def test_environmental_keywords(sentence, expected):
    assert is_environmental(sentence) is expected, environmental_hits(sentence)


@pytest.fixture
def lm(tmp_path):
    p = tmp_path / "lm.csv"
    pd.DataFrame({
        "Word": ["ACHIEVED", "IMPROVE", "STRENGTHEN", "LEADERSHIP", "LOSS", "ADVERSE", "OLDPOS"],
        "Negative": [0, 0, 0, 0, 2009, 2009, 0],
        "Positive": [2009, 2009, 2009, 2009, 0, 0, -2020],  # OLDPOS was removed
    }).to_csv(p, index=False)
    return LMDictionary.from_csv(p)


def test_lm_dictionary_and_tone(lm):
    assert "OLDPOS" not in lm.positive
    assert sentence_tone(GREEN[0], lm) == "positive"
    assert sentence_tone(NEUTRAL_ENV[0], lm) == "neutral"
    assert sentence_tone("Emissions caused an adverse loss.", lm) == "negative"


def test_lm_negation(lm):
    assert tone_counts("We have not achieved our emissions target.", lm) == (0, 0)
    assert tone_counts("We never did improve the site.", lm) == (0, 0)
    assert tone_counts("Not only did we cut emissions, we also achieved our goal.", lm) == (1, 0)


def test_mda_in_financial_section_and_roman_item_one():
    from fixtures import make_10k_mda_elsewhere

    parsed = parse_filing(make_10k_mda_elsewhere(GREEN))
    assert parsed.source == "items"
    assert parsed.item7_source == "mda_heading"
    assert len(parsed.sections["item1"].split()) > 1000  # "ITEM I. BUSINESS"
    assert "greenhouse gas emissions" in parsed.sections["item7"]
    assert "We audited the statements" not in parsed.sections["item7"]


def test_mda_from_exhibit13():
    from fixtures import make_10k_mda_elsewhere, make_exhibit13
    from talk.parse import add_exhibit13_mda

    parsed = parse_filing(make_10k_mda_elsewhere(GREEN, exhibit13=True))
    assert parsed.item7_source == "missing" and parsed.item7_short
    assert add_exhibit13_mda(parsed, make_exhibit13(GREEN))
    assert parsed.item7_source == "ex13" and not parsed.item7_short
    assert "greenhouse gas emissions" in parsed.sections["item7"]
    assert "Letter to shareholders" not in parsed.sections["item7"]
    assert not add_exhibit13_mda(parse_filing(make_10k_mda_elsewhere(GREEN, True)), "<p>no mda</p>")


def test_no_item_headings_uses_fulltext():
    body = "<p>MANAGEMENT'S DISCUSSION AND ANALYSIS</p><p>" + FILLER_TEXT + "</p><p>Consolidated Statements of Income</p>"
    parsed = parse_filing(f"<html><body>{body}</body></html>")
    assert parsed.source == "fulltext" and parsed.item7_source == "fulltext"


FILLER_TEXT = "We sell products to customers worldwide and grow revenue every year. " * 300


@pytest.mark.parametrize("sentence", [
    "We expand the partner ecosystem for our payment network products.",
    "A significant adverse change in the business climate could affect the asset.",
    "The SCB replaced the CET1 capital conservation buffer of 2.5%.",
    "Patches may not exist given our networked environment and many systems.",
    "Employees maintain a respectful environment at every site we operate.",
])
def test_business_uses_are_not_environmental(sentence):
    assert not is_environmental(sentence), environmental_hits(sentence)
