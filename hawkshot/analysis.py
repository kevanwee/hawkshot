from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class ProfileComputation:
    distances_m: np.ndarray
    terrain_m: np.ndarray
    terrain_with_curvature_m: np.ndarray
    sight_line_m: np.ndarray
    curvature_bulge_m: np.ndarray


@dataclass(frozen=True)
class VisibilityResult:
    visible: bool
    obstruction_index: Optional[int]
    obstruction_distance_m: Optional[float]
    clearance_at_obstruction_m: Optional[float]
    observer_elevation_m: float
    target_elevation_m: float
    total_distance_m: float
    profile: ProfileComputation


def evaluate_profile_visibility(
    terrain_profile_m: Sequence[float],
    total_distance_m: float,
    observer_height_m: float = 0.0,
    target_height_m: float = 0.0,
    refraction_coeff: float = 0.13,
    apply_curvature: bool = True,
    obstruction_tolerance_m: float = 0.0,
) -> VisibilityResult:
    """
    Evaluate LOS visibility along a sampled terrain profile.

    Parameters
    ----------
    terrain_profile_m:
        Elevation profile sampled from observer to target (meters above sea level).
    total_distance_m:
        Horizontal distance from observer to target in meters.
    observer_height_m, target_height_m:
        Height above local terrain for observer/target in meters.
    refraction_coeff:
        Atmospheric refraction coefficient k in [0, 1). Typical value is ~0.13.
    apply_curvature:
        Whether to include Earth curvature/refraction correction.
    obstruction_tolerance_m:
        Required clearance margin in meters. Values > 0 are more conservative.
    """
    terrain = np.asarray(terrain_profile_m, dtype=float)
    if terrain.ndim != 1:
        raise ValueError("terrain_profile_m must be a 1D sequence.")
    if terrain.size < 2:
        raise ValueError("terrain_profile_m must contain at least two samples.")
    if total_distance_m <= 0:
        raise ValueError("total_distance_m must be greater than 0.")
    if not (0 <= refraction_coeff < 1):
        raise ValueError("refraction_coeff must be in [0, 1).")

    distances_m = np.linspace(0.0, float(total_distance_m), terrain.size)

    observer_elevation_m = float(terrain[0] + observer_height_m)
    target_elevation_m = float(terrain[-1] + target_height_m)
    sight_line_m = observer_elevation_m + (
        (target_elevation_m - observer_elevation_m) * (distances_m / total_distance_m)
    )

    if apply_curvature:
        effective_radius_m = EARTH_RADIUS_M / (1.0 - refraction_coeff)
        curvature_bulge_m = (distances_m * (total_distance_m - distances_m)) / (
            2.0 * effective_radius_m
        )
    else:
        curvature_bulge_m = np.zeros_like(distances_m)

    terrain_with_curvature_m = terrain + curvature_bulge_m
    excess_m = terrain_with_curvature_m - sight_line_m

    interior_excess = excess_m[1:-1]
    obstruction_candidates = np.where(interior_excess > obstruction_tolerance_m)[0]

    obstruction_index: Optional[int]
    obstruction_distance_m: Optional[float]
    clearance_at_obstruction_m: Optional[float]

    if obstruction_candidates.size:
        obstruction_index = int(obstruction_candidates[0] + 1)
        obstruction_distance_m = float(distances_m[obstruction_index])
        clearance_at_obstruction_m = float(excess_m[obstruction_index])
        visible = False
    else:
        obstruction_index = None
        obstruction_distance_m = None
        clearance_at_obstruction_m = None
        visible = True

    profile = ProfileComputation(
        distances_m=distances_m,
        terrain_m=terrain,
        terrain_with_curvature_m=terrain_with_curvature_m,
        sight_line_m=sight_line_m,
        curvature_bulge_m=curvature_bulge_m,
    )
    return VisibilityResult(
        visible=visible,
        obstruction_index=obstruction_index,
        obstruction_distance_m=obstruction_distance_m,
        clearance_at_obstruction_m=clearance_at_obstruction_m,
        observer_elevation_m=observer_elevation_m,
        target_elevation_m=target_elevation_m,
        total_distance_m=float(total_distance_m),
        profile=profile,
    )
