"""
Trading calendar module for dynamic EIA release date computation.

The EIA Natural Gas Weekly Storage Report is released every Thursday at 10:30 AM ET.
When Monday of that week is a market holiday (CMEGlobex_NatGas holiday schedule), the release
shifts to Friday at 10:30 AM ET.

This module NEVER hard-codes "Thursday" — it always derives the release date dynamically.
"""
import logging
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
EIA_RELEASE_TIME = time(10, 30)  # 10:30 AM ET


def get_eia_release_date(reference_date: date = None) -> date:
    """
    Compute the EIA storage report release date for the week containing *reference_date*.

    Standard schedule: Thursday 10:30 AM ET.
    Holiday shift: If Monday of that week is a market holiday, release shifts to Friday.

    Args:
        reference_date: Any date within the target week. Defaults to today.

    Returns:
        The date on which the EIA report is released that week.
    """
    if reference_date is None:
        reference_date = date.today()

    # Find Monday of the week containing reference_date
    monday = reference_date - timedelta(days=reference_date.weekday())
    thursday = monday + timedelta(days=3)
    friday = monday + timedelta(days=4)

    # Use CMEGlobex_NatGas calendar (most relevant for natural gas futures trading)
    cme = mcal.get_calendar("CMEGlobex_NatGas")
    # Check if Monday is a holiday (not a trading day)
    schedule = cme.schedule(start_date=monday.isoformat(), end_date=monday.isoformat())
    monday_is_holiday = schedule.empty

    if monday_is_holiday:
        logger.info(
            "Monday %s is a CME holiday — EIA release shifts to Friday %s",
            monday.isoformat(),
            friday.isoformat(),
        )
        return friday
    else:
        return thursday


def get_eia_release_datetime(reference_date: date = None) -> datetime:
    """
    Return the full timezone-aware EIA release datetime (10:30 AM ET) for the given week.
    """
    release_date = get_eia_release_date(reference_date)
    return datetime.combine(release_date, EIA_RELEASE_TIME, tzinfo=ET)


def get_week_release_dates(start_date: date, end_date: date) -> list[date]:
    """
    Return a list of EIA release dates for all weeks between start_date and end_date.

    Args:
        start_date: First week's Monday (or any date in that week).
        end_date: Last week's Monday (or any date in that week).

    Returns:
        Sorted list of EIA release dates.
    """
    dates = []
    current = start_date - timedelta(days=start_date.weekday())  # Monday of start week
    last_monday = end_date - timedelta(days=end_date.weekday())   # Monday of end week

    while current <= last_monday:
        release = get_eia_release_date(current)
        dates.append(release)
        current += timedelta(weeks=1)

    return sorted(set(dates))


def is_eia_release_day(check_date: date = None) -> bool:
    """Return True if check_date is the EIA release day for its week."""
    if check_date is None:
        check_date = date.today()
    return get_eia_release_date(check_date) == check_date


def get_report_week_ending(release_date: date) -> date:
    """
    Given an EIA release date, return the week-ending date the report covers.
    The EIA storage report covers the week ending the Friday before release.

    For a Thursday release: covers week ending previous Friday (6 days prior).
    For a Friday release (holiday week): covers week ending same-week Friday... 
    Actually EIA reports week ending Friday before the release Thursday.
    Week ending date = the Friday immediately before the release Thursday/Friday.
    """
    # Release is Thursday or Friday
    # The report covers the week ending on the Friday of the PRIOR week
    # i.e., 6 days before Thursday, or 7 days before Friday
    weekday = release_date.weekday()  # Thursday=3, Friday=4
    if weekday == 3:  # Thursday
        return release_date - timedelta(days=6)  # Previous Friday
    elif weekday == 4:  # Friday
        return release_date - timedelta(days=7)  # Previous Friday
    else:
        # Fallback: previous Friday
        days_to_prev_friday = (weekday - 4) % 7
        return release_date - timedelta(days=days_to_prev_friday)


def get_next_eia_release() -> datetime:
    """Return the next upcoming EIA release datetime from today."""
    today = date.today()
    release_this_week = get_eia_release_date(today)
    if release_this_week >= today:
        return get_eia_release_datetime(today)
    # Already past this week's release — return next week's
    next_week = today + timedelta(weeks=1)
    return get_eia_release_datetime(next_week)
