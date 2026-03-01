"""Hawkshot: line-of-sight and viewshed utilities for DEM data."""

from .analysis import (
    EARTH_RADIUS_M,
    ProfileComputation,
    VisibilityResult,
    evaluate_profile_visibility,
)

__all__ = [
    "EARTH_RADIUS_M",
    "ProfileComputation",
    "VisibilityResult",
    "evaluate_profile_visibility",
]
