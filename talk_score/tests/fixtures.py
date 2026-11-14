"""Synthetic 10-K documents and a fake EDGAR client for offline tests."""
from __future__ import annotations

import pandas as pd

FILLER = (
    "The Company designs, manufactures and sells products to customers in many markets. "
    "Revenue depends on demand from customers and on general economic conditions. "
)


def make_10k(env_sentences: list[str], filler_repeats: int = 120, ixbrl: bool = False) -> str:
    """HTML 10-K with a table of contents, Items 1, 1A, 1B, 2, 7, 7A, 8,
    a numeric table, a cross reference and (optionally) hidden iXBRL facts."""
    filler = FILLER * filler_repeats
    env = " ".join(env_sentences)
    hidden = "<ix:header><ix:hidden>HIDDEN XBRL FACT emissions renewable</ix:hidden></ix:header>" if ixbrl else ""
    return f"""<html><head><title>10-K</title><style>p {{color: red}}</style></head><body>
{hidden}
<div>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</div>
<table>
<tr><td>Item 1.</td><td>Business</td><td>3</td></tr>
<tr><td>Item 1A.</td><td>Risk Factors</td><td>10</td></tr>
<tr><td>Item 1B.</td><td>Unresolved Staff Comments</td><td>20</td></tr>
<tr><td>Item 2.</td><td>Properties</td><td>21</td></tr>
<tr><td>Item 7.</td><td>Management's Discussion and Analysis</td><td>30</td></tr>
<tr><td>Item 7A.</td><td>Quantitative and Qualitative Disclosures</td><td>40</td></tr>
<tr><td>Item 8.</td><td>Financial Statements</td><td>41</td></tr>
</table>
<p style="font-weight:bold"><span>ITEM 1.</span>&nbsp;<span>BUSINESS</span></p>
<p>{filler}</p>
<p>{env}</p>
<p style="font-weight:bold">ITEM 1A. RISK FACTORS</p>
<p>Investors should carefully consider the risks below. {filler}</p>
<p>For more information see Item 7, Management's Discussion and Analysis, below.</p>
<p>ITEM 1B. UNRESOLVED STAFF COMMENTS</p><p>None.</p>
<p>ITEM 2. PROPERTIES</p><p>We own our headquarters.</p>
<p>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION</p>
<p>{filler}</p>
<table><tr><td>Revenue</td><td>1,234.5</td><td>1,100.2</td></tr><tr><td>Net income</td><td>234.1</td><td>210.9</td></tr></table>
<p>ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK</p><p>Interest rate risk is limited.</p>
<p>ITEM 8. FINANCIAL STATEMENTS</p><p>See the audited statements. {filler}</p>
</body></html>"""


def make_10k_mda_elsewhere(env_sentences: list[str], exhibit13: bool = False) -> str:
    """Item 7 only refers elsewhere: to a Financial Section later in the same
    document (Exxon style), or to Exhibit 13 (IBM / Wells Fargo style)."""
    filler = FILLER * 120
    env = " ".join(env_sentences)
    where = "Refer to the 2015 Annual Report to Stockholders, incorporated herein by reference." if exhibit13 \
        else "Reference is made to the section in the Financial Section of this report."
    financial_section = "" if exhibit13 else f"""
<p>FINANCIAL SECTION</p>
<p>MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS</p>
<p>{filler}</p><p>{env}</p>
<p>MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS</p>
<p>{filler}</p>
<p>REPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM</p><p>We audited the statements. {filler}</p>"""
    return f"""<html><body>
<p>ITEM I. BUSINESS</p><p>{filler}</p>
<p>ITEM 1A. RISK FACTORS</p><p>{filler}</p>
<p>ITEM 2. PROPERTIES</p><p>Offices.</p>
<p>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</p><p>{where}</p>
<p>ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA</p><p>See the Financial Section.</p>
{financial_section}
</body></html>"""


def make_exhibit13(env_sentences: list[str]) -> str:
    return f"""<html><body><p>Letter to shareholders: we love the planet and achieved everything.</p>
<p>Management Discussion</p><p>{FILLER * 60}</p><p>{" ".join(env_sentences)}</p>
<p>Report of Independent Registered Public Accounting Firm</p><p>{FILLER * 20}</p></body></html>"""


EDGAR_INDEX = """<table class="tableFile" summary="Document Format Files">
<tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
<tr><td>1</td><td>10-K</td><td><a href="/ix?doc=/Archives/edgar/data/51143/000155837025001801/ibm-20241231x10k.htm">ibm-20241231x10k.htm</a></td><td>10-K</td><td>1</td></tr>
<tr><td>2</td><td>EX-13</td><td><a href="/Archives/edgar/data/51143/000155837025001801/ibm-20241231xex13.htm">ibm-20241231xex13.htm</a></td><td>EX-13</td><td>2</td></tr>
</table>"""


GREEN = [
    "We reduced our greenhouse gas emissions by 20 percent and achieved our renewable energy goal.",
    "Our new plants improve energy efficiency and strengthen our leadership in clean energy.",
]
NEUTRAL_ENV = ["We are subject to environmental laws governing the discharge of materials into the air."]


class FakeEdgar:
    """Stands in for EdgarClient; `docs` maps (cik, accession) -> html."""

    def __init__(self, filings: dict[int, list[dict]], docs: dict[tuple[int, str], str]):
        self.filings, self.docs = filings, docs

    def ten_k_filings(self, cik):
        df = pd.DataFrame(self.filings[cik])
        return df, {"sec_name": f"FIRM {cik}", "sic": "1000", "sic_description": "x",
                    "fiscal_year_end": "1231", "former_names": ""}

    def period_of_report(self, cik, accession):
        return None

    def exhibit(self, cik, accession, ex_type):
        return self.docs.get((cik, accession, ex_type))

    def document(self, cik, accession, doc):
        return self.docs[(cik, accession)]


class FakeClassifier:
    """Keyword stand-in for FinBERT: environmental if 'emission', 'energy' or
    'environmental' appears; positive if 'improve' or 'achieved' appears."""

    @staticmethod
    def esg(sentences):
        return ["environmental" if any(k in s.lower() for k in ("emission", "energy", "environmental"))
                else "none" for s in sentences]

    @staticmethod
    def tone(sentences):
        return ["positive" if ("improve" in s.lower() or "achieved" in s.lower()) else "neutral"
                for s in sentences]
