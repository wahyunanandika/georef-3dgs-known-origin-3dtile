# Modifications by wahyunanandika — June 2026
"""Build a similarity_transform.json from a single manual anchor point —
no Metashape camera XML required.

Use case
--------
Indoor / standalone PLY with no GPS reference at all. You pick one point in
the PLY's local space (default: bbox center) and tell this script which
real-world lat/lon/height that point should land on. The script produces
the same similarity_transform.json shape that tiles_exporter.py already
consumes (scale, rotation, translation), so the rest of the pipeline is
untouched.

What this does NOT do
----------------------
- Does not solve scale (assumes scale = 1.0, i.e. PLY units are already metres).
- Does not solve rotation from data (assumes PLY local Z = Up, and maps
  local X -> East, local Y -> North at the target point). There is no way
  to recover true orientation without ground control — if you know the
  building's real heading, rotate the PLY first (e.g. in SuperSplat) or
  extend this script with a yaw parameter.

Accuracy: this is placement, not georeferencing. Good enough for "the model
shows up roughly in the right spot in ArcGIS Pro", not for survey-grade work.

Usage
-----
python solve_manual_anchor.py splat.ply out/similarity_transform.json \
    --lat -6.92604455288287 --lon 107.6374250190141 --height 679.0 \
    [--anchor bbox_center | --anchor x,y,z]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tiles_exporter import read_ply
from transform_solver import geodetic_to_ecef


def enu_to_ecef_rotation(lat_deg: float, lon_deg: float) -> np.ndarray:
    """3x3 rotation whose columns are the East, North, Up unit vectors
    (in ECEF) at the given geodetic point."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)

    east  = np.array([-sin_lon,            cos_lon,           0.0])
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    up    = np.array([ cos_lat * cos_lon,  cos_lat * sin_lon, sin_lat])

    return np.column_stack([east, north, up])  # local X->East, Y->North, Z->Up


def bbox_center(ply_path: Path) -> np.ndarray:
    ply, _names = read_ply(ply_path)
    xyz = np.stack([ply["x"], ply["y"], ply["z"]], axis=-1).astype(np.float64)
    pmin = xyz.min(axis=0)
    pmax = xyz.max(axis=0)
    return (pmin + pmax) * 0.5


def build_manual_anchor_transform(
    anchor_local: np.ndarray,
    lat_deg: float,
    lon_deg: float,
    height_m: float,
) -> dict:
    R = enu_to_ecef_rotation(lat_deg, lon_deg)
    target_ecef = geodetic_to_ecef(lat_deg, lon_deg, height_m)
    s = 1.0
    t = target_ecef - s * (R @ anchor_local)

    return {
        "scale":       s,
        "rotation":    R.tolist(),
        "translation": t.tolist(),
        "method":      "manual_anchor",
        "anchor_local": anchor_local.tolist(),
        "anchor_target": {"lat": lat_deg, "lon": lon_deg, "height_m": height_m},
        "note": (
            "scale=1.0 assumed; rotation assumes local Z=Up, X=East, Y=North "
            "at the anchor point. No ground control was used — verify "
            "placement visually after export."
        ),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Build similarity_transform.json from a single manual anchor "
                    "(no camera XML needed).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("ply", type=Path, help="input PLY file (used for bbox center)")
    ap.add_argument("out_json", type=Path, help="output similarity_transform.json path")
    ap.add_argument("--lat", type=float, required=True, help="target latitude (deg)")
    ap.add_argument("--lon", type=float, required=True, help="target longitude (deg)")
    ap.add_argument("--height", type=float, required=True,
                    help="target height above WGS84 ellipsoid (m). "
                         "If you only have an MSL/orthometric elevation, add the "
                         "local geoid undulation N (height_ellipsoidal = height_msl + N).")
    ap.add_argument("--anchor", type=str, default="bbox_center",
                    help="'bbox_center' or 'x,y,z' local coordinates")
    args = ap.parse_args()

    if args.anchor == "bbox_center":
        anchor_local = bbox_center(args.ply)
        print(f"Anchor (PLY bbox center): {anchor_local}")
    else:
        anchor_local = np.array([float(v) for v in args.anchor.split(",")], dtype=np.float64)
        print(f"Anchor (manual): {anchor_local}")

    transform = build_manual_anchor_transform(anchor_local, args.lat, args.lon, args.height)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(transform, f, indent=2)

    print(f"Wrote {args.out_json}")
    print(f"  Target: lat={args.lat}, lon={args.lon}, height={args.height} m (ellipsoidal)")


if __name__ == "__main__":
    main()
