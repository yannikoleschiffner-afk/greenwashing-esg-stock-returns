"""Turn a 10-K main document (HTML or plain text) into clean section text.

We keep the narrative sections where firms describe themselves:
    Item 1  Business
    Item 1A Risk Factors
    Item 7  Management's Discussion and Analysis (MD&A)

Section headings also appear in the table of contents and in cross references
("see Item 1A, Risk Factors"). We therefore only accept headings at the start
of a line and, for each section, keep the start/end pair that gives the
longest section, which skips the (short) table-of-contents entries.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

BLOCK_TAGS = [
    "p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "section", "article", "ul", "ol", "center", "pre", "blockquote",
]

# Tables whose non-space characters are more than this share digits are
# financial tables, not prose.
NUMERIC_TABLE_SHARE = 0.15

# If the three items together have fewer words than this, the heading search
# failed (or the items are incorporated by reference) and we use the full text.
MIN_SECTION_WORDS = 1500

# An Item 7 shorter than this usually says "incorporated by reference to
# Exhibit 13" - flagged so it can be checked or dropped in robustness tests.
SHORT_ITEM7_WORDS = 300

# Items 1 + 1A together shorter than this: headings not found.
MIN_FRONT_WORDS = 500


def html_to_text(raw: str) -> str:
    """Strip HTML/iXBRL markup, drop numeric tables, keep paragraph breaks."""
    if not re.search(r"<(html|body|div|p|font|table)\b", raw[:20000], re.I):
        return _normalise_whitespace(raw)

    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "head", "title"]):
        tag.decompose()
    # Inline XBRL puts hidden facts in <ix:header>; lxml lowercases the name.
    for tag in soup.find_all(["ix:header", "ix:hidden"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()

    for table in soup.find_all("table"):
        text = table.get_text(" ", strip=True)
        chars = re.sub(r"\s", "", text)
        if chars and sum(c.isdigit() for c in chars) / len(chars) > NUMERIC_TABLE_SHARE:
            table.decompose()

    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    for cell in soup.find_all(["td", "th"]):
        cell.insert_after(" ")

    return _normalise_whitespace(soup.get_text(""))


def _normalise_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ").replace("​", "")
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


# --- section extraction ------------------------------------------------------

def _heading(item: str) -> re.Pattern:
    # "ITEM 1A.", "Item 1A:", "Item 1A -", "ITEM1A", "Items 1 and 2." at line start.
    # Some filers write Item 1 as "ITEM I" (HubSpot).
    num = item.replace("1", "(?:1|I)", 1) if item.startswith("1") else item
    return re.compile(
        rf"^[ \t]*items?[ \t]*{num}(?![0-9a-z])[ \t]*[\.:\-–—]?",
        re.I | re.M,
    )


SECTIONS = {
    "item1": (_heading("1"), [_heading("1a"), _heading("1b"), _heading("1c"), _heading("2")]),
    "item1a": (_heading("1a"), [_heading("1b"), _heading("1c"), _heading("2")]),
    "item7": (_heading("7"), [_heading("7a"), _heading("8")]),
}


def _longest_section(text: str, start_pat: re.Pattern, end_pats: list[re.Pattern]) -> str:
    ends = sorted(m.start() for p in end_pats for m in p.finditer(text))
    best = ""
    for m in start_pat.finditer(text):
        nxt = next((e for e in ends if e > m.end()), None)
        if nxt is None:
            continue
        chunk = text[m.start():nxt]
        if len(chunk) > len(best):
            best = chunk
    return best.strip()


# Item 7 often only says "see Management's Discussion and Analysis in the
# Financial Section" (Exxon, Chevron, JPMorgan) or points to Exhibit 13 (IBM,
# Wells Fargo). We then look for the MD&A under its own heading, ending at the
# next heading that starts the financial statements.
_MDA_START = re.compile(
    r"^[ \t]*(?:combined[ \t]+)?management[’'`]?s?[ \t]+discussion(?:[ \t]+and[ \t]+analysis)?"
    r"|^[ \t]*financial[ \t]+review[ \t]*$",
    re.I | re.M,
)
_MDA_END = [re.compile(p, re.I | re.M) for p in [
    r"^[ \t]*reports? of independent registered public accounting firm",
    r"^[ \t]*management[’'`]?s?[ \t]+(?:annual[ \t]+)?report[ \t]+on[ \t]+internal[ \t]+control",
    r"^[ \t]*consolidated[ \t]+(?:statements?|balance[ \t]+sheets?)[ \t]+of",
    r"^[ \t]*(?:item[ \t]*8\.?[ \t]*)?financial[ \t]+statements[ \t]+and[ \t]+supplementary[ \t]+data",
    r"^[ \t]*(?:item[ \t]*7a\.?[ \t]*)?quantitative[ \t]+and[ \t]+qualitative[ \t]+disclosures?",
]]


def find_mda(text: str) -> str:
    """Longest MD&A-headed block that ends at a financial-statements heading."""
    return _longest_section(text, _MDA_START, _MDA_END)


@dataclass
class ParsedFiling:
    sections: dict[str, str] = field(default_factory=dict)
    source: str = "items"  # "items" or "fulltext"
    item7_short: bool = False
    item7_source: str = "item7"  # item7 | mda_heading | ex13 | missing | fulltext

    @property
    def text(self) -> str:
        return "\n\n".join(s for s in self.sections.values() if s)

    def word_counts(self) -> dict[str, int]:
        return {f"{k}_words": len(v.split()) for k, v in self.sections.items()}


def extract_sections(text: str) -> ParsedFiling:
    sections = {name: _longest_section(text, s, e) for name, (s, e) in SECTIONS.items()}
    item7_source = "item7"
    if len(sections["item7"].split()) < SHORT_ITEM7_WORDS:
        mda = find_mda(text)
        if len(mda.split()) >= SHORT_ITEM7_WORDS:
            sections["item7"], item7_source = mda, "mda_heading"
        else:
            item7_source = "missing"
    total = sum(len(s.split()) for s in sections.values())
    front = len(sections["item1"].split()) + len(sections["item1a"].split())
    # No Item 1/1A headings at all (annual-report layouts with a cross-reference
    # index: Citi, GE, Morgan Stanley): MD&A alone would not be comparable, so
    # use the full text and flag it.
    if total < MIN_SECTION_WORDS or front < MIN_FRONT_WORDS:
        return ParsedFiling({"fulltext": text}, source="fulltext",
                            item7_short=item7_source == "missing", item7_source="fulltext")
    return ParsedFiling(sections, source="items", item7_short=item7_source == "missing",
                        item7_source=item7_source)


def add_exhibit13_mda(parsed: ParsedFiling, ex13_raw: str) -> bool:
    """Fill a missing Item 7 from the MD&A in Exhibit 13. Returns True if found."""
    mda = find_mda(html_to_text(ex13_raw))
    if len(mda.split()) < SHORT_ITEM7_WORDS:
        return False
    parsed.sections["item7"] = mda
    parsed.item7_source, parsed.item7_short = "ex13", False
    return True


def parse_filing(raw: str) -> ParsedFiling:
    return extract_sections(html_to_text(raw))


# --- sentences ---------------------------------------------------------------

_ABBREVIATIONS = [
    "U.S.A.", "U.S.", "U.K.", "E.U.", "Inc.", "Corp.", "Co.", "Ltd.", "L.P.", "LLC.",
    "No.", "Nos.", "vs.", "etc.", "e.g.", "i.e.", "Mr.", "Mrs.", "Ms.", "Dr.", "St.",
    "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Sept.", "Oct.",
    "Nov.", "Dec.", "approx.", "Fig.", "Sec.", "Rev.", "Jr.", "Sr.", "N.A.", "S.A.",
]
_ABBR_RE = re.compile("|".join(re.escape(a) for a in sorted(_ABBREVIATIONS, key=len, reverse=True)))
_SPLIT_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")

MIN_SENTENCE_WORDS = 5
MIN_ALPHA_SHARE = 0.5


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, dropping headings and table debris.

    A sentence counts only if it has at least MIN_SENTENCE_WORDS words and at
    least half of its characters are letters. The same rule defines the
    denominator of the Talk score for every method.
    """
    out: list[str] = []
    for para in re.split(r"\n+", text):
        para = para.strip()
        if not para:
            continue
        protected = _ABBR_RE.sub(lambda m: m.group(0).replace(".", "<DOT>"), para)
        for sent in _SPLIT_RE.split(protected):
            sent = sent.replace("<DOT>", ".").strip()
            if _is_prose(sent):
                out.append(sent)
    return out


def _is_prose(sent: str) -> bool:
    if len(sent.split()) < MIN_SENTENCE_WORDS:
        return False
    chars = re.sub(r"\s", "", sent)
    return bool(chars) and sum(c.isalpha() for c in chars) / len(chars) >= MIN_ALPHA_SHARE
