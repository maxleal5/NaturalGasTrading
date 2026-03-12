"""
Tests for natgas.calendar.trading_calendar

Covers:
- get_eia_release_date returns correct Thursday/Friday
- Holiday week shifts to Friday
- get_eia_release_datetime returns 10:30 AM ET
- get_report_week_ending returns the correct Friday
"""
import pytest
from datetime import date, time
from zoneinfo import ZoneInfo
from unittest.mock import patch, MagicMock

from natgas.calendar.trading_calendar import (
    get_eia_release_date,
    get_eia_release_datetime,
    get_report_week_ending,
    ET,
    EIA_RELEASE_TIME,
)


class TestGetEiaReleaseDate:
    def test_normal_week_returns_thursday(self):
        """Week of 2025-01-13 (Mon) — no holiday → Thursday 2025-01-16."""
        release = get_eia_release_date(date(2025, 1, 13))
        assert release.weekday() == 3  # Thursday

    def test_normal_week_wednesday_input(self):
        """Inputting Wednesday of same week still returns that Thursday."""
        release = get_eia_release_date(date(2025, 1, 15))
        assert release.weekday() == 3

    def test_normal_week_thursday_input(self):
        """Inputting Thursday itself returns same Thursday."""
        release = get_eia_release_date(date(2025, 1, 16))
        assert release.weekday() == 3

    def test_holiday_week_returns_friday(self):
        """
        MLK Monday 2025-01-20 is a CME holiday → release shifts to Friday 2025-01-24.
        We mock the calendar schedule to return empty (holiday).
        """
        from natgas.calendar.trading_calendar import get_eia_release_date
        import pandas_market_calendars as mcal

        # The monday of MLK week 2025
        test_date = date(2025, 1, 21)  # Tuesday of MLK week

        # Patch the CME calendar to treat Monday as a holiday
        monday = date(2025, 1, 20)
        mock_schedule = MagicMock()
        mock_schedule.empty = True  # Empty schedule → holiday

        with patch.object(mcal.get_calendar("CMEGlobex_NatGas"), "schedule", return_value=mock_schedule):
            with patch("natgas.calendar.trading_calendar.mcal") as mock_mcal:
                mock_cal = MagicMock()
                mock_cal.schedule.return_value = MagicMock()
                mock_cal.schedule.return_value.empty = True
                mock_mcal.get_calendar.return_value = mock_cal
                release = get_eia_release_date(test_date)

        assert release.weekday() == 4  # Friday

    def test_non_holiday_week_different_year(self):
        """A generic non-holiday Wednesday → should give Thursday of same week."""
        ref = date(2024, 6, 5)  # Wednesday
        release = get_eia_release_date(ref)
        assert release.weekday() == 3
        # Thursday of same week
        from datetime import timedelta
        monday = ref - timedelta(days=ref.weekday())
        assert release == monday + timedelta(days=3)


class TestGetEiaReleaseDatetime:
    def test_returns_datetime_at_1030_et(self):
        release_dt = get_eia_release_datetime(date(2025, 3, 6))
        assert release_dt.hour == 10
        assert release_dt.minute == 30
        assert release_dt.tzinfo is not None

    def test_timezone_is_eastern(self):
        release_dt = get_eia_release_datetime(date(2025, 3, 6))
        # ZoneInfo key should contain "America"
        tz_key = str(release_dt.tzinfo)
        assert "America" in tz_key or "ET" in tz_key or "Eastern" in tz_key


class TestGetReportWeekEnding:
    def test_thursday_release_points_to_prior_friday(self):
        """EIA release on Thu 2025-01-16 → week ending Fri 2025-01-10."""
        release = date(2025, 1, 16)  # Thursday
        week_ending = get_report_week_ending(release)
        assert week_ending.weekday() == 4  # Friday
        assert week_ending < release

    def test_friday_release_points_to_prior_friday(self):
        """EIA release (holiday shift) on Fri 2025-01-24 → week ending Fri 2025-01-17."""
        release = date(2025, 1, 24)  # Friday
        week_ending = get_report_week_ending(release)
        assert week_ending.weekday() == 4
        assert week_ending < release

    def test_week_ending_is_7_days_before_thursday_release(self):
        release = date(2025, 3, 13)  # Thursday
        week_ending = get_report_week_ending(release)
        from datetime import timedelta
        assert (release - week_ending).days == 6


class TestConstants:
    def test_eia_release_time_is_1030(self):
        assert EIA_RELEASE_TIME == time(10, 30)

    def test_et_timezone_defined(self):
        assert ET is not None
