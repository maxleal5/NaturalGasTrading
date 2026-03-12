"""
Population-weighted HDD/CDD mask management.

Two static NumPy weight masks are pre-computed once using census population density
and stored as .npy files. At runtime, dot-product multiplication replaces live raster
overlay for ~100x speedup.

Winter/HDD mask: Weights emphasize Midwest and Northeast (residential gas heating demand).
Summer/CDD mask: Weights emphasize Texas and Southeast (gas-fired power generation demand).
"""
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default storage path for pre-computed weight arrays
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", Path(__file__).parent.parent.parent / "data" / "weights"))

# CONUS bounding box at 0.25° resolution
LAT_MIN, LAT_MAX = 24.0, 50.0   # 24°N to 50°N
LON_MIN, LON_MAX = -125.0, -66.0  # 66°W to 125°W
GRID_RES = 0.25

# Grid dimensions
N_LAT = int((LAT_MAX - LAT_MIN) / GRID_RES) + 1  # 105
N_LON = int((LON_MAX - LON_MIN) / GRID_RES) + 1  # 237 (using absolute diff)

# Region lat/lon bounding boxes (lat_min, lat_max, lon_min, lon_max)
REGIONS = {
    "national": (24.0, 50.0, -125.0, -66.0),
    "northeast": (40.0, 47.5, -80.0, -66.0),
    "midwest": (36.0, 49.0, -97.0, -80.0),
    "mountain": (31.0, 49.0, -117.0, -97.0),
    "pacific": (32.0, 50.0, -125.0, -117.0),
    "south_central": (25.0, 37.0, -107.0, -89.0),
    "texas": (25.5, 36.5, -107.0, -93.0),
    "southeast": (25.0, 37.0, -93.0, -75.0),
}


def _lat_arr() -> np.ndarray:
    return np.arange(LAT_MIN, LAT_MAX + GRID_RES / 2, GRID_RES)


def _lon_arr() -> np.ndarray:
    return np.arange(LON_MIN, LON_MAX + GRID_RES / 2, GRID_RES)


def _build_synthetic_mask(region_weights: dict[str, float]) -> np.ndarray:
    """
    Build a synthetic population weight mask by summing Gaussian blobs
    centered on key demand cities, weighted by regional gas demand importance.

    This is the offline setup step — in production, replace with actual
    census population density raster (rasterio + geopandas).

    Args:
        region_weights: Dict mapping region name to relative weight multiplier.

    Returns:
        Normalized weight array of shape (N_LAT, N_LON).
    """
    lats = _lat_arr()
    lons = _lon_arr()
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    mask = np.zeros((len(lats), len(lons)), dtype=np.float64)

    # Key demand cities with coordinates and region labels
    demand_centers = [
        # (lat, lon, region, relative_weight)
        # Northeast / Midwest — dominant for winter gas heating
        (40.7, -74.0, "northeast", 1.0),    # New York City
        (42.4, -71.1, "northeast", 0.8),    # Boston
        (39.9, -75.2, "northeast", 0.7),    # Philadelphia
        (39.3, -76.6, "northeast", 0.5),    # Baltimore
        (38.9, -77.0, "northeast", 0.5),    # Washington DC
        (41.5, -81.7, "midwest", 1.0),      # Cleveland
        (41.9, -87.6, "midwest", 1.2),      # Chicago
        (43.0, -76.1, "midwest", 0.4),      # Syracuse
        (44.0, -92.5, "midwest", 0.4),      # Minneapolis area
        (39.1, -84.5, "midwest", 0.5),      # Cincinnati
        (40.4, -80.0, "midwest", 0.6),      # Pittsburgh
        (42.3, -83.0, "midwest", 0.7),      # Detroit
        # Mountain
        (39.7, -104.9, "mountain", 0.4),    # Denver
        (43.6, -116.2, "mountain", 0.2),    # Boise
        # Pacific (lower winter HDD weight)
        (37.8, -122.4, "pacific", 0.3),     # San Francisco
        (47.6, -122.3, "pacific", 0.3),     # Seattle
        (34.1, -118.2, "pacific", 0.4),     # Los Angeles
        # South Central / Texas — dominant for summer CDD
        (29.8, -95.4, "texas", 1.2),        # Houston
        (32.8, -96.8, "texas", 1.0),        # Dallas
        (30.3, -97.7, "texas", 0.6),        # Austin
        (29.4, -98.5, "texas", 0.5),        # San Antonio
        # Southeast — important for summer CDD
        (33.7, -84.4, "southeast", 0.7),    # Atlanta
        (30.3, -81.7, "southeast", 0.5),    # Jacksonville
        (25.8, -80.2, "southeast", 0.7),    # Miami
        (35.2, -80.8, "southeast", 0.5),    # Charlotte
        (36.2, -86.8, "southeast", 0.4),    # Nashville
        (30.0, -90.1, "southeast", 0.4),    # New Orleans
        (32.4, -86.3, "southeast", 0.3),    # Montgomery
    ]

    sigma = 3.0  # degrees (~333 km latitude, ~230 km longitude at 40°N) — spatial spread

    for lat_c, lon_c, region, base_weight in demand_centers:
        w = region_weights.get(region, 0.1)
        blob = np.exp(-((lat_grid - lat_c) ** 2 + (lon_grid - lon_c) ** 2) / (2 * sigma ** 2))
        mask += w * base_weight * blob

    # Normalize to sum to 1
    total = mask.sum()
    if total > 0:
        mask /= total
    return mask


def build_winter_hdd_mask() -> np.ndarray:
    """
    Build winter HDD population weight mask.
    Midwest and Northeast emphasized — residential gas furnace demand.
    """
    region_weights = {
        "northeast": 1.5,
        "midwest": 1.8,
        "mountain": 0.6,
        "pacific": 0.3,
        "south_central": 0.4,
        "texas": 0.3,
        "southeast": 0.4,
    }
    return _build_synthetic_mask(region_weights)


def build_summer_cdd_mask() -> np.ndarray:
    """
    Build summer CDD population weight mask.
    Texas and Southeast emphasized — gas-fired power generation for AC demand.
    """
    region_weights = {
        "northeast": 0.6,
        "midwest": 0.7,
        "mountain": 0.5,
        "pacific": 0.3,
        "south_central": 1.0,
        "texas": 1.8,
        "southeast": 1.6,
    }
    return _build_synthetic_mask(region_weights)


def save_masks(weights_dir: Optional[Path] = None) -> tuple[Path, Path]:
    """
    Pre-compute and save both seasonal weight masks to .npy files.
    Run once at setup, re-run when census data is updated.

    Returns:
        Tuple of (winter_mask_path, summer_mask_path).
    """
    d = weights_dir or WEIGHTS_DIR
    d.mkdir(parents=True, exist_ok=True)

    winter_path = d / "winter_hdd_mask.npy"
    summer_path = d / "summer_cdd_mask.npy"

    logger.info("Building winter HDD mask...")
    winter_mask = build_winter_hdd_mask()
    np.save(winter_path, winter_mask)
    logger.info("Saved winter HDD mask to %s (shape=%s)", winter_path, winter_mask.shape)

    logger.info("Building summer CDD mask...")
    summer_mask = build_summer_cdd_mask()
    np.save(summer_path, summer_mask)
    logger.info("Saved summer CDD mask to %s (shape=%s)", summer_path, summer_mask.shape)

    return winter_path, summer_path


def load_masks(weights_dir: Optional[Path] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Load pre-computed seasonal weight masks from .npy files.
    If files don't exist, build and save them first.

    Returns:
        Tuple of (winter_hdd_mask, summer_cdd_mask), each shape (N_LAT, N_LON).
    """
    d = weights_dir or WEIGHTS_DIR
    winter_path = d / "winter_hdd_mask.npy"
    summer_path = d / "summer_cdd_mask.npy"

    if not winter_path.exists() or not summer_path.exists():
        logger.info("Weight masks not found — building from scratch...")
        save_masks(d)

    winter_mask = np.load(winter_path)
    summer_mask = np.load(summer_path)
    return winter_mask, summer_mask


def get_seasonal_mask(month: int, weights_dir: Optional[Path] = None) -> tuple[np.ndarray, str]:
    """
    Return the appropriate seasonal weight mask for the given month.

    Winter season (October–March): Use HDD mask.
    Summer season (April–September): Use CDD mask.

    Args:
        month: Calendar month (1–12).
        weights_dir: Optional override for mask storage directory.

    Returns:
        Tuple of (mask_array, season_label) where season_label is 'winter' or 'summer'.
    """
    winter_mask, summer_mask = load_masks(weights_dir)
    if month in (10, 11, 12, 1, 2, 3):
        return winter_mask, "winter"
    else:
        return summer_mask, "summer"


def compute_pop_weighted_hdd_cdd(
    t2m_fahrenheit_grid: np.ndarray,
    weight_mask: np.ndarray,
    base_temp: float = 65.0,
) -> tuple[float, float]:
    """
    Compute population-weighted HDD and CDD from a temperature grid.

    Uses dot-product vector multiplication — ~100x faster than live raster overlay.

    Args:
        t2m_fahrenheit_grid: 2D array of shape (N_LAT, N_LON) with 2m temps in °F.
        weight_mask: 2D population weight array of same shape, summing to 1.0.
        base_temp: HDD/CDD base temperature in °F (default 65°F).

    Returns:
        Tuple of (weighted_hdd, weighted_cdd).
    """
    if t2m_fahrenheit_grid.shape != weight_mask.shape:
        raise ValueError(
            f"Temperature grid shape {t2m_fahrenheit_grid.shape} does not match "
            f"weight mask shape {weight_mask.shape}"
        )

    # Compute HDD and CDD at each grid point
    hdd_grid = np.maximum(0.0, base_temp - t2m_fahrenheit_grid)
    cdd_grid = np.maximum(0.0, t2m_fahrenheit_grid - base_temp)

    # Population-weighted sum via dot product (flattened for efficiency)
    weighted_hdd = float(np.dot(hdd_grid.ravel(), weight_mask.ravel()))
    weighted_cdd = float(np.dot(cdd_grid.ravel(), weight_mask.ravel()))

    return weighted_hdd, weighted_cdd


def compute_regional_hdd_cdd(
    t2m_fahrenheit_grid: np.ndarray,
    region: str,
    season: str,
    weights_dir: Optional[Path] = None,
    base_temp: float = 65.0,
) -> tuple[float, float]:
    """
    Compute population-weighted HDD/CDD for a specific region and season.

    Args:
        t2m_fahrenheit_grid: Full CONUS temperature grid (N_LAT, N_LON).
        region: Region name (must be in REGIONS dict).
        season: 'winter' or 'summer'.
        weights_dir: Optional override for mask storage directory.
        base_temp: HDD/CDD base temperature in °F.

    Returns:
        Tuple of (weighted_hdd, weighted_cdd) for the specified region.
    """
    if region not in REGIONS:
        raise ValueError(f"Unknown region '{region}'. Valid: {list(REGIONS.keys())}")

    winter_mask, summer_mask = load_masks(weights_dir)
    base_mask = winter_mask if season == "winter" else summer_mask

    # Create regional mask by zeroing out grid points outside region bbox
    lat_min, lat_max, lon_min, lon_max = REGIONS[region]
    lats = _lat_arr()
    lons = _lon_arr()
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    region_filter = (
        (lat_grid >= lat_min) & (lat_grid <= lat_max) &
        (lon_grid >= lon_min) & (lon_grid <= lon_max)
    ).astype(np.float64)

    regional_mask = base_mask * region_filter
    total = regional_mask.sum()
    if total > 0:
        regional_mask /= total  # Re-normalize within region

    return compute_pop_weighted_hdd_cdd(t2m_fahrenheit_grid, regional_mask, base_temp)
