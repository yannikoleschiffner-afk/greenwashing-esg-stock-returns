"""Minimal SEC EDGAR client.

We download only each 10-K's main document, not the full submission, which
also contains every exhibit and the XBRL files. That cuts the download from
roughly 100 GB to a few GB for the S&P 500 over 2010-2024. Filing date and
period of report come straight from EDGAR's filing index.

SEC rules: identify yourself in the User-Agent ("Name email@domain") and stay
below 10 requests per second.
"""
from __future__ import annotations

import re
import time

import pandas as pd
import requests

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
CIK_LOOKUP_URL = "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"

TEN_K_FORMS = {"10-K", "10-K405"}


class EdgarClient:
    def __init__(self, user_agent: str, max_per_second: float = 8.0, retries: int = 5):
        if "@" not in user_agent:
            raise ValueError('SEC requires a User-Agent like "Jane Doe jane@uni.edu"')
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.min_interval = 1.0 / max_per_second
        self.retries = retries
        self._last = 0.0

    def get(self, url: str) -> requests.Response:
        r = None
        for attempt in range(self.retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.session.get(url, timeout=60)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                r.raise_for_status()
            # 403/429 = rate limited, 5xx = server trouble: back off and retry
            time.sleep(2 ** (attempt + 1))
        if r is not None:
            r.raise_for_status()
        raise requests.HTTPError(f"GET {url} failed after {self.retries} attempts")

    # --- firm identifiers -----------------------------------------------------

    def company_tickers(self) -> pd.DataFrame:
        """Current ticker -> CIK map (listed firms only)."""
        data = self.get(TICKERS_URL).json()
        df = pd.DataFrame(data.values()).rename(columns={"cik_str": "cik", "title": "sec_name"})
        return df[["cik", "ticker", "sec_name"]]

    def cik_lookup(self) -> pd.DataFrame:
        """Every EDGAR entity name (current and former) with its CIK."""
        return parse_cik_lookup(self.get(CIK_LOOKUP_URL).content.decode("latin-1"))

    # --- filings ---------------------------------------------------------------

    def submissions(self, cik: int) -> dict:
        return self.get(SUBMISSIONS_URL.format(cik=int(cik))).json()

    def ten_k_filings(self, cik: int) -> tuple[pd.DataFrame, dict]:
        """All original 10-K filings of a firm (amendments excluded)."""
        sub = self.submissions(cik)
        frames = [pd.DataFrame(sub["filings"]["recent"])]
        for page in sub["filings"].get("files", []):
            frames.append(pd.DataFrame(self.get(SUBMISSIONS_PAGE_URL.format(name=page["name"])).json()))
        info = {
            "sec_name": sub.get("name"),
            "sic": sub.get("sic"),
            "sic_description": sub.get("sicDescription"),
            "fiscal_year_end": sub.get("fiscalYearEnd"),
            "former_names": "; ".join(f["name"] for f in sub.get("formerNames", [])),
        }
        return filter_ten_k(pd.concat(frames, ignore_index=True)), info

    def document(self, cik: int, accession: str, doc: str) -> str:
        url = ARCHIVE_URL.format(cik=int(cik), acc_nodash=accession.replace("-", ""), doc=doc)
        r = self.get(url)
        return r.content.decode(r.encoding or "utf-8", errors="replace")

    def exhibit(self, cik: int, accession: str, ex_type: str = "EX-13") -> str | None:
        """Text of an exhibit (e.g. EX-13, the annual report to shareholders)."""
        index = self.document(cik, accession, f"{accession}-index.htm")
        url = find_exhibit_url(index, ex_type)
        if url is None:
            return None
        r = self.get(url)
        return r.content.decode(r.encoding or "utf-8", errors="replace")

    def period_of_report(self, cik: int, accession: str) -> str | None:
        """Fallback when the index has no reportDate: read the SGML header."""
        doc = f"{accession}.hdr.sgml"
        try:
            text = self.document(cik, accession, doc)
        except requests.HTTPError:
            return None
        return parse_period(text)


def filter_ten_k(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["accessionNumber", "filingDate", "reportDate", "form", "primaryDocument"]
    df = df[df["form"].isin(TEN_K_FORMS)][cols].copy()
    df.columns = ["accession", "filing_date", "period_of_report", "form", "primary_document"]
    df["period_of_report"] = df["period_of_report"].replace("", None)
    return df.reset_index(drop=True)


def find_exhibit_url(index_html: str, ex_type: str) -> str | None:
    """Link to the first document of type `ex_type` (EX-13, EX-13.1, ...) in
    a filing's -index.htm page."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(index_html, "lxml")
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        link = row.find("a", href=True)
        if len(cells) < 4 or link is None:
            continue
        doc_type = cells[3].get_text(strip=True).upper()
        if doc_type == ex_type or doc_type.startswith(ex_type + "."):
            href = link["href"].replace("/ix?doc=", "")
            if not href.lower().endswith((".htm", ".html", ".txt")):
                continue
            return href if href.startswith("http") else "https://www.sec.gov" + href
    return None


def parse_period(header: str) -> str | None:
    m = re.search(r"(?:<PERIOD>|CONFORMED PERIOD OF REPORT:)\s*(\d{8})", header)
    if not m:
        return None
    d = m.group(1)
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


def parse_cik_lookup(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        m = re.match(r"^(.*):(\d{10}):\s*$", line)
        if m:
            rows.append((m.group(1).strip(), int(m.group(2))))
    return pd.DataFrame(rows, columns=["sec_name", "cik"])
