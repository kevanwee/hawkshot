import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin

from hawkshot.dem import DEMData, sample_profile_from_lonlat
from hawkshot.workflow import analyze_viewshed


def _build_synthetic_dem() -> DEMData:
    array = np.zeros((220, 220), dtype=float)
    transform = from_origin(-1.0, 1.0, 0.01, 0.01)
    return DEMData(
        name="synthetic",
        array=array,
        transform=transform,
        crs=CRS.from_epsg(4326),
        nodata=None,
    )


def test_profile_sampling_hits_expected_endpoints() -> None:
    dem = _build_synthetic_dem()
    sampled = sample_profile_from_lonlat(
        dem=dem,
        observer_lonlat=(-0.5, 0.5),
        target_lonlat=(0.5, -0.5),
        sample_spacing_m=10_000.0,
    )
    assert sampled.rows[0] == 50
    assert sampled.cols[0] == 50
    assert sampled.rows[-1] == 150
    assert sampled.cols[-1] == 150
    assert sampled.total_distance_m > 0


def test_workflow_visible_on_flat_dem() -> None:
    dem = _build_synthetic_dem()
    analysis = analyze_viewshed(
        dem=dem,
        observer_lonlat=(-0.5, 0.5),
        target_lonlat=(0.5, -0.5),
        observer_height_m=2.0,
        target_height_m=2.0,
        sample_spacing_m=10_000.0,
        apply_curvature=False,
    )
    assert analysis.result.visible is True
