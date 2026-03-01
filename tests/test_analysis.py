import numpy as np
import pytest

from hawkshot.analysis import evaluate_profile_visibility


def test_visible_on_flat_terrain_without_curvature() -> None:
    terrain = np.zeros(25)
    result = evaluate_profile_visibility(
        terrain_profile_m=terrain,
        total_distance_m=1000.0,
        observer_height_m=2.0,
        target_height_m=2.0,
        apply_curvature=False,
    )
    assert result.visible is True
    assert result.obstruction_index is None


def test_blocked_by_midpoint_ridge() -> None:
    terrain = np.zeros(11)
    terrain[5] = 30.0
    result = evaluate_profile_visibility(
        terrain_profile_m=terrain,
        total_distance_m=1000.0,
        observer_height_m=1.0,
        target_height_m=1.0,
        apply_curvature=False,
    )
    assert result.visible is False
    assert result.obstruction_index == 5
    assert result.clearance_at_obstruction_m is not None
    assert result.clearance_at_obstruction_m > 0


def test_curvature_can_block_long_flat_path() -> None:
    terrain = np.zeros(201)
    without_curvature = evaluate_profile_visibility(
        terrain_profile_m=terrain,
        total_distance_m=50_000.0,
        observer_height_m=2.0,
        target_height_m=2.0,
        apply_curvature=False,
    )
    with_curvature = evaluate_profile_visibility(
        terrain_profile_m=terrain,
        total_distance_m=50_000.0,
        observer_height_m=2.0,
        target_height_m=2.0,
        apply_curvature=True,
        refraction_coeff=0.13,
    )
    assert without_curvature.visible is True
    assert with_curvature.visible is False


def test_refraction_reduces_curvature_bulge() -> None:
    terrain = np.zeros(101)
    no_refraction = evaluate_profile_visibility(
        terrain_profile_m=terrain,
        total_distance_m=30_000.0,
        observer_height_m=2.0,
        target_height_m=2.0,
        apply_curvature=True,
        refraction_coeff=0.0,
    )
    standard_refraction = evaluate_profile_visibility(
        terrain_profile_m=terrain,
        total_distance_m=30_000.0,
        observer_height_m=2.0,
        target_height_m=2.0,
        apply_curvature=True,
        refraction_coeff=0.13,
    )
    peak_no_refraction = float(np.max(no_refraction.profile.curvature_bulge_m))
    peak_standard_refraction = float(np.max(standard_refraction.profile.curvature_bulge_m))
    assert peak_standard_refraction < peak_no_refraction


def test_rejects_invalid_distance() -> None:
    with pytest.raises(ValueError):
        evaluate_profile_visibility([0.0, 0.0], total_distance_m=0.0)
