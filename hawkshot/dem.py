from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from pyproj import Geod, Transformer
import rasterio
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import Affine, rowcol


@dataclass(frozen=True)
class DEMData:
    name: str
    array: np.ndarray
    transform: Affine
    crs: Optional[CRS]
    nodata: Optional[float]


@dataclass(frozen=True)
class SampledProfile:
    total_distance_m: float
    lons: np.ndarray
    lats: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    terrain_m: np.ndarray


def load_dem(path: str | Path) -> DEMData:
    dem_path = Path(path)
    with rasterio.open(dem_path) as ds:
        array = ds.read(1).astype(float)
        return DEMData(
            name=str(dem_path),
            array=array,
            transform=ds.transform,
            crs=ds.crs,
            nodata=ds.nodata,
        )


def load_dem_bytes(data: bytes, name: str = "uploaded_dem") -> DEMData:
    with MemoryFile(data) as memfile:
        with memfile.open() as ds:
            array = ds.read(1).astype(float)
            return DEMData(
                name=name,
                array=array,
                transform=ds.transform,
                crs=ds.crs,
                nodata=ds.nodata,
            )


def lonlat_to_rowcol(dem: DEMData, lon: float, lat: float) -> Tuple[int, int]:
    x, y = _transform_lonlat_to_dem_crs(dem, np.array([lon]), np.array([lat]))
    row, col = rowcol(dem.transform, x[0], y[0], op=np.round)
    return int(row), int(col)


def sample_profile_from_lonlat(
    dem: DEMData,
    observer_lonlat: Sequence[float],
    target_lonlat: Sequence[float],
    sample_spacing_m: float = 30.0,
) -> SampledProfile:
    if sample_spacing_m <= 0:
        raise ValueError("sample_spacing_m must be greater than 0.")

    observer_lon, observer_lat = float(observer_lonlat[0]), float(observer_lonlat[1])
    target_lon, target_lat = float(target_lonlat[0]), float(target_lonlat[1])

    geod = Geod(ellps="WGS84")
    _, _, total_distance_m = geod.inv(observer_lon, observer_lat, target_lon, target_lat)
    if total_distance_m <= 0:
        raise ValueError("Observer and target coordinates must be different.")

    num_samples = max(2, int(np.ceil(total_distance_m / sample_spacing_m)) + 1)

    if num_samples > 2:
        mid_points = geod.npts(
            observer_lon, observer_lat, target_lon, target_lat, npts=num_samples - 2
        )
        all_points = [(observer_lon, observer_lat), *mid_points, (target_lon, target_lat)]
    else:
        all_points = [(observer_lon, observer_lat), (target_lon, target_lat)]

    lons = np.array([p[0] for p in all_points], dtype=float)
    lats = np.array([p[1] for p in all_points], dtype=float)

    xs, ys = _transform_lonlat_to_dem_crs(dem, lons, lats)
    rows, cols = rowcol(dem.transform, xs, ys, op=np.round)
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)

    if not _indices_in_bounds(dem.array, rows, cols):
        first_bad = np.where(
            (rows < 0)
            | (rows >= dem.array.shape[0])
            | (cols < 0)
            | (cols >= dem.array.shape[1])
        )[0][0]
        raise ValueError(
            "Sample point is outside DEM bounds at "
            f"index {int(first_bad)} (lon={lons[first_bad]:.6f}, lat={lats[first_bad]:.6f})."
        )

    terrain_m = dem.array[rows, cols].astype(float)
    _validate_terrain_samples(terrain_m, dem.nodata)

    return SampledProfile(
        total_distance_m=float(total_distance_m),
        lons=lons,
        lats=lats,
        rows=rows,
        cols=cols,
        terrain_m=terrain_m,
    )


def _transform_lonlat_to_dem_crs(
    dem: DEMData, lons: np.ndarray, lats: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if dem.crs is None:
        return lons, lats

    if dem.crs.is_geographic and dem.crs.to_epsg() == 4326:
        return lons, lats

    transformer = Transformer.from_crs("EPSG:4326", dem.crs, always_xy=True)
    x, y = transformer.transform(lons, lats)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _indices_in_bounds(array: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> bool:
    return bool(
        np.all((rows >= 0) & (rows < array.shape[0]) & (cols >= 0) & (cols < array.shape[1]))
    )


def _validate_terrain_samples(terrain_m: np.ndarray, nodata: Optional[float]) -> None:
    if np.any(np.isnan(terrain_m)):
        raise ValueError("DEM sampling produced NaN values; check input raster.")
    if nodata is None:
        return

    if np.any(np.isclose(terrain_m, nodata)):
        raise ValueError("Sample profile intersects DEM nodata cells.")
