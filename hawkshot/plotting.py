from __future__ import annotations

import matplotlib.pyplot as plt

from .analysis import VisibilityResult
from .dem import DEMData, SampledProfile


def build_viewshed_figure(
    dem: DEMData,
    sampled: SampledProfile,
    result: VisibilityResult,
) -> plt.Figure:
    fig, (ax_map, ax_profile) = plt.subplots(
        2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [2, 1]}
    )

    image = ax_map.imshow(dem.array, cmap="terrain")
    fig.colorbar(image, ax=ax_map, label="Elevation (m)", shrink=0.8)

    obs_col, obs_row = sampled.cols[0], sampled.rows[0]
    tgt_col, tgt_row = sampled.cols[-1], sampled.rows[-1]
    ax_map.plot(obs_col, obs_row, "g*", markersize=12, label="Observer")
    ax_map.plot(tgt_col, tgt_row, "r^", markersize=10, label="Target")

    if result.visible:
        ax_map.plot(sampled.cols, sampled.rows, "g-", linewidth=2, label="Clear LOS")
        ax_map.set_title("Line of Sight: Target Visible")
    else:
        obstruction_idx = int(result.obstruction_index)  # guaranteed by visible=False
        ax_map.plot(
            sampled.cols[: obstruction_idx + 1],
            sampled.rows[: obstruction_idx + 1],
            "g-",
            linewidth=2,
            label="Visible segment",
        )
        ax_map.plot(
            sampled.cols[obstruction_idx:],
            sampled.rows[obstruction_idx:],
            "r--",
            linewidth=2,
            label="Obstructed segment",
        )
        ax_map.plot(
            sampled.cols[obstruction_idx],
            sampled.rows[obstruction_idx],
            "yo",
            markersize=9,
            label="First obstruction",
        )
        ax_map.set_title("Line of Sight: Target Not Visible")

    ax_map.set_xlabel("Column")
    ax_map.set_ylabel("Row")
    ax_map.grid(alpha=0.25)
    ax_map.legend(loc="upper right")

    distances_km = result.profile.distances_m / 1000.0
    ax_profile.fill_between(
        distances_km, 0, result.profile.terrain_m, color="#5d8a54", alpha=0.45, label="Terrain"
    )
    ax_profile.plot(
        distances_km,
        result.profile.terrain_with_curvature_m,
        color="#2f5f2f",
        linewidth=1.2,
        linestyle="--",
        label="Terrain + curvature",
    )
    ax_profile.plot(
        distances_km, result.profile.sight_line_m, color="#1f77b4", linewidth=2, label="LOS line"
    )

    ax_profile.plot(
        distances_km[0],
        result.profile.sight_line_m[0],
        "g*",
        markersize=11,
        label="Observer point",
    )
    ax_profile.plot(
        distances_km[-1],
        result.profile.sight_line_m[-1],
        "r^",
        markersize=9,
        label="Target point",
    )

    if not result.visible:
        obstruction_idx = int(result.obstruction_index)
        ax_profile.plot(
            distances_km[obstruction_idx],
            result.profile.terrain_with_curvature_m[obstruction_idx],
            "ro",
            markersize=8,
            label="Obstruction",
        )

    ax_profile.set_title("Elevation Profile")
    ax_profile.set_xlabel("Distance (km)")
    ax_profile.set_ylabel("Elevation (m)")
    ax_profile.set_ylim(bottom=0)
    ax_profile.grid(alpha=0.25)
    ax_profile.legend(loc="best")

    fig.tight_layout()
    return fig
