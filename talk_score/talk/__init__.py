"""Talk score: positive environmental language in 10-K filings.

Pipeline stages (see README.md):
    firms     -> map the group's firm list to SEC CIKs
    download  -> fetch 10-K main documents from EDGAR and extract Items 1, 1A, 7
    score     -> Loughran-McDonald (robustness) and FinBERT (main) sentence scoring
    panel     -> one row per firm-year with the raw Talk measures
    percentile-> Talk percentile within year x 2-digit TRBC industry
"""
