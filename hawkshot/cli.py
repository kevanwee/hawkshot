from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .dem import load_dem
from .plotting import build_viewshed_figure
from .workflow import analyze_viewshed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hawkshot line-of-sight/viewshed analysis")
    parser.add_argument("--dem", required=True, help="Path to DEM raster (GeoTIFF, DTED, etc.)")
    parser.add_argument("--observer-lon", required=True, type=float)
    parser.add_argument("--observer-lat", required=True, type=float)
    parser.add_argument("--target-lon", required=True, type=float)
    parser.add_argument("--target-lat", required=True, type=float)
    parser.add_argument("--observer-height-m", type=float, default=0.0)
    parser.add_argument("--target-height-m", type=float, default=0.0)
    parser.add_argument("--sample-spacing-m", type=float, default=30.0)
    parser.add_argument("--refraction-coeff", type=float, default=0.13)
    parser.add_argument(
        "--obstruction-tolerance-m",
        type=float,
        default=0.0,
        help="How much clearance is required above terrain (meters).",
    )
    parser.add_argument(
        "--no-curvature",
        action="store_true",
        help="Disable Earth curvature/refraction correction.",
    )
    parser.add_argument("--plot", action="store_true", help="Display matplotlib figure.")
    parser.add_argument(
        "--output-plot",
        default=None,
        help="Optional output image path (e.g. output.png).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dem = load_dem(args.dem)

    analysis = analyze_viewshed(
        dem=dem,
        observer_lonlat=(args.observer_lon, args.observer_lat),
        target_lonlat=(args.target_lon, args.target_lat),
        observer_height_m=args.observer_height_m,
        target_height_m=args.target_height_m,
        sample_spacing_m=args.sample_spacing_m,
        refraction_coeff=args.refraction_coeff,
        apply_curvature=not args.no_curvature,
        obstruction_tolerance_m=args.obstruction_tolerance_m,
    )
    result = analysis.result

    print("===== Hawkshot Visibility Analysis =====")
    print(f"DEM: {dem.name}")
    print(f"Samples: {len(result.profile.distances_m)}")
    print(f"Total distance: {result.total_distance_m / 1000.0:.3f} km")
    print(f"Observer effective elevation: {result.observer_elevation_m:.2f} m")
    print(f"Target effective elevation: {result.target_elevation_m:.2f} m")

    if result.visible:
        print("Result: VISIBLE")
    else:
        idx = int(result.obstruction_index)
        print("Result: NOT VISIBLE")
        print(f"First obstruction sample index: {idx}")
        print(f"Obstruction distance: {result.obstruction_distance_m / 1000.0:.3f} km")
        print(f"Required additional clearance: {result.clearance_at_obstruction_m:.2f} m")

    should_render_figure = args.plot or args.output_plot is not None
    if should_render_figure:
        fig = build_viewshed_figure(dem, analysis.sampled, result)
        if args.output_plot:
            out_path = Path(args.output_plot)
            fig.savefig(out_path, dpi=150)
            print(f"Saved plot: {out_path}")
        if args.plot:
            plt.show()
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
