"""Fiscal-year labelling shared by the whole group.

Rule agreed for the project:
    * `year` = calendar year in which the fiscal year ends,
      EXCEPT fiscal years ending in January (or in the first week of February,
      which covers 52/53-week retail calendars such as "Saturday nearest
      January 31"). Those belong to the previous calendar year, because almost
      all of the fiscal year falls there.
    * Returns are matched on calendar year `year + 1`, i.e. we assume the
      fiscal-year-`year` 10-K is known on 1 January of `year + 1`.

Examples
    Apple      FYE 2015-09-26 -> 2015 -> returns 2016
    Microsoft  FYE 2015-06-30 -> 2015 -> returns 2016
    Target     FYE 2016-01-30 -> 2015 -> returns 2016
    Walmart    FYE 2016-01-31 -> 2015 (Walmart itself calls this "fiscal 2016")
    Kellogg    FYE 2016-01-02 -> 2015 (52/53-week year ending near Dec 31)
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

# Fiscal years ending on or before this day of February count as "January" year-ends.
FEB_CUTOFF_DAY = 7


def assign_year(period_end: dt.date | pd.Timestamp | str) -> int:
    """Return the project `year` for a fiscal year ending on `period_end`."""
    d = pd.Timestamp(period_end)
    if d.month == 1 or (d.month == 2 and d.day <= FEB_CUTOFF_DAY):
        return d.year - 1
    return d.year


def assign_year_series(period_end: pd.Series) -> pd.Series:
    d = pd.to_datetime(period_end)
    shift = (d.dt.month == 1) | ((d.dt.month == 2) & (d.dt.day <= FEB_CUTOFF_DAY))
    return (d.dt.year - shift.astype(int)).astype("Int64")
