from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from hawkshot.dem import load_dem, load_dem_bytes
from hawkshot.plotting import build_viewshed_figure
from hawkshot.workflow import analyze_viewshed


st.set_page_config(page_title="Hawkshot Viewshed", layout="wide")
st.title("Hawkshot Viewshed Analysis")
st.caption("Analyze line-of-sight visibility between two geospatial points over a DEM.")

source_mode = st.radio("DEM source", ["Upload file", "Local path"], horizontal=True)
dem = None

if source_mode == "Upload file":
    uploaded = st.file_uploader(
        "Upload DEM",
        type=["tif", "tiff", "dt0", "dt1", "dt2", "hgt", "img"],
        help="Supported if rasterio can read the format.",
    )
    if uploaded is not None:
        dem = load_dem_bytes(uploaded.getvalue(), name=uploaded.name)
else:
    default_path = "dted.dt2"
    dem_path = st.text_input("DEM path", value=default_path)
    if dem_path:
        try:
            dem = load_dem(dem_path)
        except Exception as ex:
            st.error(f"Unable to load DEM path: {ex}")

col1, col2 = st.columns(2)
with col1:
    observer_lon = st.number_input("Observer longitude", value=-115.68148, format="%.6f")
    observer_lat = st.number_input("Observer latitude", value=42.93113, format="%.6f")
    observer_height_m = st.number_input(
        "Observer height above ground (m)", value=0.0, min_value=0.0, step=1.0
    )
with col2:
    target_lon = st.number_input("Target longitude", value=-115.71633, format="%.6f")
    target_lat = st.number_input("Target latitude", value=42.85085, format="%.6f")
    target_height_m = st.number_input(
        "Target height above ground (m)", value=0.0, min_value=0.0, step=1.0
    )

col3, col4, col5 = st.columns(3)
with col3:
    sample_spacing_m = st.number_input(
        "Sample spacing (m)", value=30.0, min_value=1.0, step=1.0
    )
with col4:
    apply_curvature = st.checkbox("Apply Earth curvature/refraction", value=True)
with col5:
    refraction_coeff = st.number_input(
        "Refraction coefficient (k)",
        value=0.13,
        min_value=0.0,
        max_value=0.99,
        step=0.01,
        format="%.2f",
    )

if st.button("Run Analysis", type="primary", use_container_width=True):
    if dem is None:
        st.error("Load a DEM first.")
    else:
        try:
            analysis = analyze_viewshed(
                dem=dem,
                observer_lonlat=(observer_lon, observer_lat),
                target_lonlat=(target_lon, target_lat),
                observer_height_m=observer_height_m,
                target_height_m=target_height_m,
                sample_spacing_m=sample_spacing_m,
                refraction_coeff=refraction_coeff,
                apply_curvature=apply_curvature,
            )
            result = analysis.result

            summary = st.columns(4)
            summary[0].metric("Distance (km)", f"{result.total_distance_m / 1000.0:.3f}")
            summary[1].metric("Samples", f"{len(result.profile.distances_m)}")
            summary[2].metric("Observer elev (m)", f"{result.observer_elevation_m:.2f}")
            summary[3].metric("Target elev (m)", f"{result.target_elevation_m:.2f}")

            if result.visible:
                st.success("Target is visible from observer location.")
            else:
                st.error("Target is not visible from observer location.")
                st.write(
                    f"First obstruction at {result.obstruction_distance_m / 1000.0:.3f} km. "
                    f"Missing clearance: {result.clearance_at_obstruction_m:.2f} m."
                )

            fig = build_viewshed_figure(dem, analysis.sampled, result)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as ex:
            st.error(f"Analysis failed: {ex}")
