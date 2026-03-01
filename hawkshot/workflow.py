from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .analysis import VisibilityResult, evaluate_profile_visibility
from .dem import DEMData, SampledProfile, sample_profile_from_lonlat


@dataclass(frozen=True)
class ViewshedAnalysis:
    sampled: SampledProfile
    result: VisibilityResult


def analyze_viewshed(
    dem: DEMData,
    observer_lonlat: Sequence[float],
    target_lonlat: Sequence[float],
    observer_height_m: float = 0.0,
    target_height_m: float = 0.0,
    sample_spacing_m: float = 30.0,
    refraction_coeff: float = 0.13,
    apply_curvature: bool = True,
    obstruction_tolerance_m: float = 0.0,
) -> ViewshedAnalysis:
    sampled = sample_profile_from_lonlat(
        dem=dem,
        observer_lonlat=observer_lonlat,
        target_lonlat=target_lonlat,
        sample_spacing_m=sample_spacing_m,
    )
    result = evaluate_profile_visibility(
        terrain_profile_m=sampled.terrain_m,
        total_distance_m=sampled.total_distance_m,
        observer_height_m=observer_height_m,
        target_height_m=target_height_m,
        refraction_coeff=refraction_coeff,
        apply_curvature=apply_curvature,
        obstruction_tolerance_m=obstruction_tolerance_m,
    )
    return ViewshedAnalysis(sampled=sampled, result=result)
