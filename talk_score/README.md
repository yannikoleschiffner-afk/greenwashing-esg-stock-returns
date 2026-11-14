# Talk score: positive environmental language in 10-Ks

This code produces the **Talk** half of the Greenwashing Index (GWI = Talk − Walk) for
S&P 500 firms, fiscal years 2010–2024. The output is one row per firm-year, ready to
merge with the Walk data and the return data for the regression.

## What is measured

For every 10-K we read **Item 1 (Business), Item 1A (Risk Factors) and Item 7 (MD&A)**,
split them into sentences, and count sentences that are *environmental* and *positive*:

| Measure | Environmental? | Positive? | Role |
|---|---|---|---|
| `fb_*` FinBERT | FinBERT-ESG label = Environmental | FinBERT-tone label = Positive | **main** |
| `lm_*` Loughran-McDonald | environmental keyword list (`talk/keywords.py`) | more LM positive than LM negative words (negation-adjusted) | robustness |

```
talk_raw     = positive environmental sentences / all sentences
             = env_share            x  env_pos_share
               (how much they talk     (how positively they talk
                about the environment)  about it)
```

The score is a **share**, not a count. 10-Ks grew a lot longer over 2010–2024 and big
firms write longer reports, so a count would mostly pick up size and time.

**Talk percentile** (`talk_pct`, `talk_pct_lm`): the firm's rank within its
**year × 2-digit TRBC industry** group, from 0 (least green talk) to 100 (most), with
`pct = (rank − 1)/(n − 1) × 100` and ties sharing the average rank. This is the same
scale as the Walk percentile, so `GWI = talk_pct − (100 − walk_pct)` works as in the
proposal. Standardise GWI afterwards, as the proposal says.

Only sentences with at least five words and at least 50% letters count as sentences.
This drops headings and leftover table rows. FinBERT only sees sentences that match
the environmental keyword list (the "prefilter"); see *Checks* below for how to verify
that this does not miss much.

## Conventions the whole group must share

**Year.** `year` = the calendar year in which the fiscal year ends. The exception is
fiscal years ending in January or in the first week of February (retailers like Walmart,
Target and Home Depot, and Nvidia): they count as the *previous* year. For example:

| Firm | Fiscal year ends | `year` |
|---|---|---|
| most firms | 2015-12-31 | 2015 |
| Apple | 2015-09-26 | 2015 |
| Microsoft | 2015-06-30 | 2015 |
| Target / Walmart | 2016-01-30 / 2016-01-31 | 2015 (Walmart itself calls it "fiscal 2016") |

Do not use firms' own fiscal-year labels: Walmart and Target label the same period
differently.

**Returns.** Talk for `year` t is matched with the calendar-year return of t+1
(`return_year` column). We assume each 10-K is known on 1 January of t+1. In reality
most 10-Ks come out between January and March, so the first 2–3 months overlap.
Mention this as a limitation. A good robustness check is April(t+1)–March(t+2) returns
for all firms.

**Sample.** Firm-year (i, t) is in the sample if firm i was in the S&P 500 on
**31 December of year t**, including firms that later left the index (no survivorship
bias). Firms that leave the index or are delisted during t+1 stay in with their return
up to the delisting date.

**Share classes.** GOOGL/GOOG, FOXA/FOX, NWSA/NWS, etc. are one company with one 10-K.
They get the same Talk value; `secondary_share_class = True` marks the rows to drop in
the regression.

## Input: the firm list (from Datastream)

One file for the whole group, so that Talk, Walk and returns cover the same firm-years.
Pull the historical S&P 500 constituent list for December of each year 2010–2024.
Datastream stores these as the list mnemonic plus month and year, e.g. `LS&PCOMP1215`
for December 2015. Check the exact mnemonic in the Datastream navigator. Firms that
were later delisted are included. Save one row per firm and year:

| column | example | needed? |
|---|---|---|
| `isin` (or Datastream code) | US0378331005 | yes, used as `firm_id` |
| `name` | APPLE | yes, for matching dead firms |
| `ticker` | @AAPL or AAPL | strongly recommended |
| `year` | 2015 | yes |
| `trbc` (TRBC code, e.g. 5710101010, or the economic sector) | 5710101010 | for percentiles |

Column names are detected automatically (`isin`/`dscd`/`firm_id`, `name`/`company`,
`ticker`/`symbol`, `year`); otherwise pass `--id-col`, `--name-col` and so on.

**Starting before the Datastream list arrives:** `universe-github` builds the same
firm-year list from the public ticker history at
[fja05680/sp500](https://github.com/fja05680/sp500). It adds names and CIKs from
Wikipedia's "List of S&P 500 companies":
* current members come with their CIK;
* removed firms get their name from the table of index changes;
* ticker renames (FB → META) are detected and linked to the same company.

Firms that still can't be identified go to the manual review file. Industries are then
the **Fama-French 12** industries (from each firm's SEC SIC code, `--industry-ff12`)
until the TRBC codes arrive. Switching to the Datastream list later reuses every
download and score; only firms the public list missed are fetched.

## Running it (Google Colab recommended)

Open `notebooks/talk_score_colab.ipynb` in Colab (Runtime → Change runtime type → GPU)
and run the cells. It keeps all data on your Google Drive so a disconnect loses nothing.
Every stage can be stopped and restarted; finished work is skipped.

The same steps from a terminal, run inside `talk_score/`:

```bash
pip install -r requirements.txt
export SEC_USER_AGENT="Your Name your.email@u.nus.edu"   # SEC requires this

# 1. firm-year list: either the Datastream file, or (before it arrives) the public list
python -m talk.cli universe-github            # -> data/universe.csv

# 2. firm -> SEC CIK. Check data/firms_review.csv, add firm_id,cik rows to
#    data/firm_cik_overrides.csv (look up CIKs at https://www.sec.gov/edgar/search/)
#    and rerun until the review file is empty.
python -m talk.cli firms --universe data/universe.csv

# 3. download 10-Ks, keep Items 1/1A/7 (~7,500 filings, a few hours, ~1 GB kept)
python -m talk.cli download

# 4. score. LM: download the Master Dictionary CSV from
#    https://sraf.nd.edu/loughranmcdonald-master-dictionary/
python -m talk.cli score --method lm --lm-dict Loughran-McDonald_MasterDictionary_1993-2024.csv
python -m talk.cli score --method finbert          # GPU: ~1-2 h; CPU: a day or more

# 5. panel with percentiles
python -m talk.cli panel --universe data/universe.csv --industry-ff12       # before TRBC
python -m talk.cli panel --universe sp500_universe.csv --industry-col trbc  # with Datastream
#   (or --industry-file trbc.csv if TRBC codes come separately: firm_id[,year],trbc)
```

Test the whole chain on a few firms first with `download --limit-firms 5`.

## Output: `data/talk_panel.csv`

One row per firm-year of the universe. Talk is **missing, never 0**, when no usable
10-K was found.

| column | meaning |
|---|---|
| `firm_id`, `name`, `ticker`, `year` | from the firm list, merge keys for the other datasets |
| `return_year` | `year + 1` (the return to match) |
| `cik`, `sec_name`, `match_method` | SEC identifier and how it was matched |
| `accession`, `filing_date`, `period_of_report`, `form` | the 10-K used |
| `n_sentences` | sentences in Items 1, 1A, 7 (denominator) |
| `fb_n_env`, `fb_n_env_pos`, `fb_n_env_neg` | FinBERT environmental sentences and their tone |
| **`fb_talk_raw`** | main Talk measure (share of positive environmental sentences) |
| `fb_env_share`, `fb_env_pos_share` | the two parts of `fb_talk_raw` |
| `lm_*` | the same for the Loughran-McDonald method |
| **`talk_pct`**, `talk_pct_lm` | percentile within year × TRBC industry (0–100) |
| `industry`, `industry_year_n` | industry used (2-digit TRBC, or Fama-French 12 before TRBC is available) and how many firms were ranked in the group |
| `section_source` | `items`, or `fulltext` when no Item 1/1A headings exist (Citi, GE, Morgan Stanley use an annual-report layout) |
| `item7_source` | where the MD&A came from: `item7`; `mda_heading` (Item 7 only points to a "Financial Section" in the same document: Exxon, Chevron, JPMorgan); `ex13` (MD&A in Exhibit 13: IBM, Wells Fargo); `missing`; `fulltext` |
| `item7_short` | True when no MD&A was found (`item7_source == "missing"`) |
| `late_filer` | 10-K filed more than 90 days after fiscal year end |
| `n_10k_in_year` | >1 if the firm filed two 10-Ks mapped to the same year (fiscal year change) |
| `secondary_share_class` | drop these rows in the regression |

Other files in `data/`: `firm_cik_map.csv`, `firms_review.csv`, `firm_info.csv` (SEC
name, SIC code), `download_errors.csv`, `scores_*.csv` (per filing), and
`sentences_finbert.jsonl.gz`, which holds every candidate sentence with its FinBERT
labels and is useful for hand-checking examples in the report.

## Checks before handing the data on

1. **Coverage:** share of firm-years with `fb_talk_raw` by year; look at
   `download_errors.csv` and `firms_review.csv`.
2. **Section extraction:** share of `section_source == "fulltext"` and of each
   `item7_source` by year. Robustness: rerun the regression without the `fulltext` and
   `missing` rows.
3. **Prefilter recall:**
   `python -m talk.cli score --method finbert --no-prefilter --sample 30` scores every
   sentence of 30 filings. Compare `fb_n_env` in `scores_finbert_noprefilter.csv` with
   `scores_finbert.csv`; the prefilter should find ≥ 90% of FinBERT's environmental
   sentences.
4. **Face validity:** read 20 random sentences from `sentences_finbert.jsonl.gz` labelled
   environmental + positive. Check that utilities, energy and materials firms have a high
   `fb_env_share`.
5. **Methods agree:** correlation of `fb_talk_raw` and `lm_talk_raw` (expect clearly
   positive, not 1).

## Limitations to mention in the report

* 10-Ks are known ~2–3 months into the return year (see *Returns*).
* The MD&A is not always under Item 7; the code follows it to the Financial Section or
  to Exhibit 13 (`item7_source`). Filings without Item 1/1A headings use the full text
  (`section_source == "fulltext"`). Check that results hold without these rows.
* The Loughran-McDonald dictionary was built for general financial tone, not
  environmental text; FinBERT reads context but can be wrong on long, list-like
  sentences (it truncates at 512 tokens).
* Talk captures 10-K rhetoric only, not sustainability reports or press releases.

## Tests

`python -m pytest tests` runs the whole pipeline offline on synthetic 10-Ks, with fake
SEC and FinBERT stand-ins.

The section extraction and the keyword list were also checked on 47 real recent 10-Ks
(Apple, Exxon, Chevron, JPMorgan, Walmart, Tesla, utilities, banks, …):
* in 37, Items 1, 1A and 7 were found directly;
* in 4, the MD&A was found under its own heading;
* in 2 (IBM, Wells Fargo), the MD&A is in Exhibit 13, which the download step fetches
  (that fetch is only tested on synthetic data);
* 4 fall back to the flagged full text. With the
LM method, the highest raw Talk came from Exxon, the utilities, Chevron and Tesla, and
the lowest from banks and software firms. This is expected: Talk alone rewards heavy
emitters that talk a lot, which is why the GWI subtracts Walk.
