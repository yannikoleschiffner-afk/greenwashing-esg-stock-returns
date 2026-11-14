"""Download and scoring stages. Both can be interrupted and restarted:
finished filings are skipped.

Layout of the data directory:
    firm_cik_map.csv                  firm_id -> cik (from `firms`)
    firm_info.csv                     SEC name / SIC per cik
    filings/<cik>/<accession>.json    filing metadata (small)
    filings/<cik>/<accession>.txt.gz  extracted Items 1, 1A, 7 text
    download_errors.csv
    scores_lm.csv, scores_finbert.csv one row per filing
    sentences_finbert.jsonl.gz        FinBERT labels of candidate sentences
"""
from __future__ import annotations

import gzip
import json
import logging
import traceback
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .keywords import environmental_hits
from .parse import add_exhibit13_mda, parse_filing, split_sentences
from .years import assign_year

log = logging.getLogger(__name__)

SECTION_SEP = "\n\n=====SECTION {name}=====\n\n"


# --- download -------------------------------------------------------------------

def download(client, cik_map: pd.DataFrame, data_dir: Path, first_year: int, last_year: int) -> None:
    data_dir = Path(data_dir)
    ciks = sorted(set(cik_map["cik"].dropna().astype(int)))
    info_rows, errors = [], []
    for cik in tqdm(ciks, desc="firms"):
        try:
            filings, info = client.ten_k_filings(cik)
        except Exception as e:  # noqa: BLE001 - keep going, report at the end
            errors.append({"cik": cik, "accession": None, "error": repr(e)})
            continue
        info_rows.append({"cik": cik, **info, "n_10k_total": len(filings)})
        firm_dir = data_dir / "filings" / str(cik)

        for f in filings.itertuples(index=False):
            # Cheap pre-filter on filing date before any extra request.
            fy = int(f.filing_date[:4])
            if fy < first_year or fy > last_year + 2:
                continue
            meta_path = firm_dir / f"{f.accession}.json"
            if meta_path.exists():
                continue
            try:
                period = f.period_of_report
                if not isinstance(period, str) or not period:
                    period = client.period_of_report(cik, f.accession)
                if period is None:
                    raise ValueError("no period of report")
                year = assign_year(period)
                if not first_year <= year <= last_year:
                    continue
                raw = client.document(cik, f.accession, f.primary_document)
                parsed = parse_filing(raw)
                if parsed.item7_source == "missing":
                    try:
                        ex13 = client.exhibit(cik, f.accession, "EX-13")
                    except Exception:  # noqa: BLE001 - keep the filing, flagged as missing
                        ex13 = None
                    if ex13:
                        add_exhibit13_mda(parsed, ex13)
                firm_dir.mkdir(parents=True, exist_ok=True)
                text = "".join(SECTION_SEP.format(name=k) + v for k, v in parsed.sections.items())
                with gzip.open(firm_dir / f"{f.accession}.txt.gz", "wt", encoding="utf-8") as fh:
                    fh.write(text)
                meta = {
                    "cik": cik, "accession": f.accession, "form": f.form,
                    "filing_date": f.filing_date, "period_of_report": period, "year": year,
                    "primary_document": f.primary_document, "section_source": parsed.source,
                    "item7_short": parsed.item7_short, "item7_source": parsed.item7_source,
                    **parsed.word_counts(),
                }
                meta_path.write_text(json.dumps(meta))  # written last = filing complete
            except Exception as e:  # noqa: BLE001
                log.debug(traceback.format_exc())
                errors.append({"cik": cik, "accession": f.accession, "error": repr(e)})

    _append_csv(pd.DataFrame(info_rows), data_dir / "firm_info.csv", key="cik")
    if errors:
        _append_csv(pd.DataFrame(errors), data_dir / "download_errors.csv")
        log.warning("%d errors, see download_errors.csv", len(errors))


def load_filing_meta(data_dir: Path) -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted(Path(data_dir, "filings").glob("*/*.json"))]
    return pd.DataFrame(rows)


def read_sections(data_dir: Path, cik: int, accession: str) -> dict[str, str]:
    with gzip.open(Path(data_dir, "filings", str(cik), f"{accession}.txt.gz"), "rt", encoding="utf-8") as fh:
        text = fh.read()
    parts = text.split("\n\n=====SECTION ")
    out = {}
    for p in parts[1:]:
        name, _, body = p.partition("=====\n\n")
        out[name] = body
    return out


# --- scoring ----------------------------------------------------------------------

def filing_sentences(data_dir: Path, cik: int, accession: str) -> list[str]:
    return split_sentences("\n\n".join(read_sections(data_dir, cik, accession).values()))


def score_lm(data_dir: Path, lm) -> None:
    from .lm import sentence_tone

    out_path = Path(data_dir, "scores_lm.csv")
    todo = _todo(data_dir, out_path)
    rows = []
    for cik, acc in tqdm(todo, desc="LM"):
        sents = filing_sentences(data_dir, cik, acc)
        env = [s for s in sents if environmental_hits(s)]
        tones = [sentence_tone(s, lm) for s in env]
        rows.append({
            "accession": acc, "n_sentences": len(sents), "n_env_keyword": len(env),
            "lm_n_env": len(env), "lm_n_env_pos": tones.count("positive"),
            "lm_n_env_neg": tones.count("negative"),
        })
        if len(rows) >= 200:
            _append_csv(pd.DataFrame(rows), out_path)
            rows = []
    _append_csv(pd.DataFrame(rows), out_path)


def score_finbert(data_dir: Path, scorer, prefilter: bool = True, save_sentences: bool = True,
                  sample: int | None = None) -> None:
    """FinBERT scores. With prefilter=False every sentence goes to FinBERT
    (slow; meant for a `sample` of filings to measure the prefilter's recall,
    written to scores_finbert_noprefilter.csv)."""
    name = "scores_finbert.csv" if prefilter else "scores_finbert_noprefilter.csv"
    out_path = Path(data_dir, name)
    todo = _todo(data_dir, out_path)
    if sample:
        todo = todo[:sample]
    sent_path = Path(data_dir, "sentences_finbert.jsonl.gz" if prefilter else "sentences_finbert_noprefilter.jsonl.gz")
    rows, sent_rows = [], []
    for cik, acc in tqdm(todo, desc="FinBERT"):
        sents = filing_sentences(data_dir, cik, acc)
        hits = [environmental_hits(s) for s in sents]
        cand = [i for i, h in enumerate(hits) if h] if prefilter else list(range(len(sents)))
        labels = scorer.label([sents[i] for i in cand])
        n_env = sum(e for e, _ in labels)
        tones = [t for e, t in labels if e]
        rows.append({
            "accession": acc, "n_sentences": len(sents), "n_env_keyword": sum(bool(h) for h in hits),
            "fb_n_candidates": len(cand), "fb_n_env": n_env,
            "fb_n_env_pos": tones.count("positive"), "fb_n_env_neg": tones.count("negative"),
        })
        if save_sentences:
            for i, (e, t) in zip(cand, labels):
                if e or hits[i]:
                    sent_rows.append({"accession": acc, "sentence": sents[i], "keywords": hits[i],
                                      "fb_env": e, "fb_tone": t})
        if len(rows) >= 50:
            _flush(rows, out_path, sent_rows, sent_path)
            rows, sent_rows = [], []
    _flush(rows, out_path, sent_rows, sent_path)


def _flush(rows, out_path, sent_rows, sent_path):
    # Sentences first: if we crash in between, the filing is simply redone.
    if sent_rows:
        with gzip.open(sent_path, "at", encoding="utf-8") as fh:
            for r in sent_rows:
                fh.write(json.dumps(r) + "\n")
    _append_csv(pd.DataFrame(rows), out_path)


def _todo(data_dir: Path, out_path: Path) -> list[tuple[int, str]]:
    meta = load_filing_meta(data_dir)
    if meta.empty:
        return []
    done = set(pd.read_csv(out_path, usecols=["accession"])["accession"]) if out_path.exists() else set()
    meta = meta[~meta["accession"].isin(done)]
    return list(zip(meta["cik"].astype(int), meta["accession"]))


def _append_csv(df: pd.DataFrame, path: Path, key: str | None = None) -> None:
    if df.empty:
        return
    path = Path(path)
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
        if key:
            df = df.drop_duplicates(key, keep="last")
    df.to_csv(path, index=False)
