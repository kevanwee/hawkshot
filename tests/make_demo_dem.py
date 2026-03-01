from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def build_demo_dem(path: Path) -> None:
    width, height = 500, 500
    pixel_deg = 0.0004
    lon_min = -116.0
    lat_max = 43.1

    x = np.linspace(-1.0, 1.0, width)
    y = np.linspace(-1.0, 1.0, height)
    xx, yy = np.meshgrid(x, y)

    base = 1400.0 + 150.0 * np.sin(2.5 * xx) + 130.0 * np.cos(2.0 * yy)
    ridge = 300.0 * np.exp(-((xx - 0.1) ** 2 / 0.02 + (yy + 0.2) ** 2 / 0.06))
    crater = -180.0 * np.exp(-((xx + 0.45) ** 2 / 0.05 + (yy - 0.3) ** 2 / 0.04))
    terrain = (base + ridge + crater).astype(np.float32)

    transform = from_origin(lon_min, lat_max, pixel_deg, pixel_deg)

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as ds:
        ds.write(terrain, 1)


if __name__ == "__main__":
    output = Path("tests/data/demo_dem.tif")
    build_demo_dem(output)
    print(f"Created demo DEM at {output.resolve()}")
