# 3dtiles_georeference_3dgs

Place a 3D Gaussian Splatting (3DGS) scene on the globe and export it as **3D Tiles 1.1** with **SPZ-compressed Gaussian splats**, viewable in **CesiumJS / Cesium ion** and **ArcGIS Pro**.

The pipeline takes a standard 3DGS `.ply`, a 7-parameter similarity transform (scale, rotation, translation) from the PLY's local frame to ECEF, and writes an octree-tiled tileset where every tile is a GLB carrying an SPZ v3 payload.

> Adapted from [dozeri83/geo-register-plugin](https://github.com/dozeri83/geo-register-plugin) (GPL-3.0), reworked into a standalone Python pipeline.

---

## Contents

- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Pipeline overview](#pipeline-overview)
- [Input PLY requirements](#input-ply-requirements)
- [Step 1 — Build the similarity transform](#step-1--build-the-similarity-transform)
  - [Option A: Manual anchor (known origin, e.g. a benchmark)](#option-a-manual-anchor-known-origin-eg-a-benchmark)
  - [Option B: Camera-based solve (Python API)](#option-b-camera-based-solve-python-api)
- [Step 2 — Export 3D Tiles](#step-2--export-3d-tiles)
- [Step 3 — Check the tileset](#step-3--check-the-tileset)
- [Step 4 (optional) — Package as `.3tz`](#step-4-optional--package-as-3tz)
- [Viewing the result](#viewing-the-result)
- [Output format details](#output-format-details)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Repository layout

| File | Role |
|---|---|
| `solve_manual_anchor.py` | CLI. Builds `similarity_transform.json` from one known anchor point (no camera data needed). |
| `tiles_exporter.py` | CLI + module. Reads the PLY, builds the octree, writes `tileset.json` and `tile_XXXX.glb`. |
| `spz_encode.py` | Module. Pure-Python SPZ v3 encoder (mirrors Niantic's `load-spz.cc` v3 path). |
| `transform_solver.py` | Module. WGS84 geodetic ⇄ ECEF conversion, Umeyama, RANSAC-Umeyama, and the camera-based `solve_ply_to_ecef` solver. |
| `sanity_check.ipynb` | Notebook. Prints the extensions declared in a generated `tileset.json`. |

---

## Requirements

- Python 3 (tested on 3.11)
- `numpy`
- Node.js — only for the optional `.3tz` packaging step

```bash
pip install numpy
```

All scripts import each other by module name, so run them from the repository root.

---

## Pipeline overview

```
splat.ply (local frame)
      │
      ▼
[1] similarity_transform.json      ← solve_manual_anchor.py  (or transform_solver API)
      │   scale · rotation · translation   (local → ECEF)
      ▼
[2] tiles_exporter.py
      │
      ▼
out_dir/
├── tileset.json                   ← root.transform = similarity matrix
├── tile_0000.glb                  ← SPZ-compressed splats, local coordinates
├── tile_0001.glb
└── ...
      │
      ▼
[3] sanity_check.ipynb  (optional)
[4] npx 3d-tiles-tools convert → out.3tz  (optional)
```

Splat positions inside each GLB stay in the PLY's local frame. Georeferencing is carried entirely by the root tile's `transform` matrix, which keeps vertex coordinates small and float32-safe.

---

## Input PLY requirements

`tiles_exporter.py` reads standard 3DGS PLYs with these constraints:

- `format binary_little_endian`
- **every** vertex property is `float` (float32)
- required properties:

| Property | Meaning |
|---|---|
| `x`, `y`, `z` | position |
| `rot_0` … `rot_3` | rotation quaternion, **w x y z** order |
| `scale_0` … `scale_2` | log-scale |
| `opacity` | opacity logit |
| `f_dc_0` … `f_dc_2` | SH degree-0 color |
| `f_rest_*` | optional higher-order SH (channel-major, as written by standard 3DGS trainers) |

If `f_rest_*` is missing or has fewer coefficients than `--max-sh-degree` needs, the exporter falls back to SH degree 0.

---

## Step 1 — Build the similarity transform

The exporter consumes a JSON file with this shape:

```json
{
  "scale": 1.0,
  "rotation": [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]],
  "translation": [tx, ty, tz]
}
```

It maps a local PLY point `p` to ECEF as `p_ecef = scale · R · p + t`. The short keys `s`, `R`, `t` are also accepted. Extra keys are ignored.

### Option A: Manual anchor (known origin, e.g. a benchmark)

`solve_manual_anchor.py` places the PLY on the globe using one point whose real-world position you know.

> **Use this only when the PLY's origin `(0,0,0)` corresponds to a known coordinate**, for example a benchmark (BM) or a surveyed control point that the scan was set up on. Without a known origin, the result is only approximate placement, not georeferencing.

**Assumptions built into the script**

- `scale = 1.0` → PLY units must already be metres.
- Rotation is not solved. The local axes are mapped as **X → East, Y → North, Z → Up** at the anchor. If the scene is not already oriented that way, rotate the PLY first (e.g. in SuperSplat).
- `--height` is **ellipsoidal** (WGS84).

**Usage**

```bash
python solve_manual_anchor.py <splat.ply> <out/similarity_transform.json> \
    --lat <deg> --lon <deg> --height <ellipsoidal_m> \
    [--anchor bbox_center | --anchor x,y,z]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `ply` | yes | | Input PLY (only used when `--anchor bbox_center`) |
| `out_json` | yes | | Output `similarity_transform.json` path (folders created automatically) |
| `--lat` | yes | | Target latitude, decimal degrees |
| `--lon` | yes | | Target longitude, decimal degrees |
| `--height` | yes | | Target height above the WGS84 ellipsoid, metres |
| `--anchor` | no | `bbox_center` | Local point that lands on the target: `bbox_center` or `x,y,z` |

**Typical case: origin sits on a benchmark**

```bash
python solve_manual_anchor.py scan.ply out/similarity_transform.json \
    --lat -6.92604455288287 --lon 107.6374250190141 --height 679.0 \
    --anchor 0,0,0
```

**Converting a BM elevation to ellipsoidal height**

Benchmark elevations are usually orthometric (above the geoid / MSL). Convert before passing `--height`:

```
h (ellipsoidal) = H (orthometric) + N (geoid undulation)
```

Take `N` at the BM location from a geoid model (e.g. EGM2008 via [GeographicLib](https://geographiclib.sourceforge.io/C++/doc/geoid.html), or a national model where available). Passing an orthometric height directly will place the model off by `N` vertically.

**Note on SuperSplat.** SuperSplat rotates around the origin `(0,0,0)`. After rotating, the origin stays put but the bbox center moves, so keep `--anchor 0,0,0` rather than `bbox_center`.

The output JSON also records `method`, `anchor_local`, `anchor_target` and a `note` for traceability.

### Option B: Camera-based solve (Python API)

`transform_solver.py` provides a full similarity solve from matched camera positions. There is no CLI for it in this repo; call it from Python.

```python
import json
import numpy as np
from transform_solver import solve_ply_to_ecef

# {image_stem: camera_center_in_PLY_frame}, e.g. COLMAP C = -R.T @ T
colmap_cameras = {
    "DJI_0001": np.array([x, y, z]),
    # ...
}

# Metashape-style camera list: lat/lon/alt (ellipsoidal) or precomputed ECEF
metashape_data = {
    "cameras": [
        {"name": "DJI_0001", "lat": -6.89, "lon": 107.61, "alt": 780.0},
        # or {"name": "...", "ecef": [X, Y, Z]}
    ]
}

result = solve_ply_to_ecef(colmap_cameras, metashape_data,
                           ransac_inlier_thr_m=5.0)

with open("out/similarity_transform.json", "w") as f:
    json.dump(result, f, indent=2)
```

- Cameras are matched by name (`colmap_cameras` key ↔ `cameras[i]["name"]`); at least 4 matches are required.
- Solver: RANSAC around Umeyama (4-point samples, default inlier threshold 5 m), then a refit on inliers.
- Returns `scale`, `rotation`, `translation`, `rmse_m`, `n_inliers`, `n_total`, and prints the camera centroid in lat/lon/alt.
- Optional `points3d` (N×3 sparse points in the PLY frame) prints the transformed terrain altitude range as a sanity check.

Other helpers in the module: `geodetic_to_ecef`, `ecef_to_geodetic`, `umeyama`, `ransac_umeyama`.

---

## Step 2 — Export 3D Tiles

```bash
python tiles_exporter.py <splat.ply> <similarity_transform.json> <out_dir> [options]
```

| Argument | Default | Description |
|---|---|---|
| `ply` | | Input PLY |
| `similarity_json` | | Transform from Step 1 |
| `out_dir` | | Output folder (created if missing) |
| `--max-sh-degree {0,1,2,3}` | `3` | Highest SH degree to include |
| `--max-splats-per-tile INT` | `0` (auto) | Max splats per leaf tile |
| `--min-tile-size FLOAT` | `0.1` | Stop splitting when a cell diagonal is below this (PLY units) |
| `--fraction FLOAT` | `1.0` | Random subsample for quick tests, `0 < f ≤ 1` |
| `--seed INT` | `0` | RNG seed for `--fraction` |

**Auto tile size** (when `--max-splats-per-tile 0`):

| Total splats | Max splats per tile |
|---|---|
| < 500k | 20,000 |
| < 2M | 30,000 |
| < 10M | 50,000 |
| < 20M | 75,000 |
| ≥ 20M | 100,000 |

**Examples**

```bash
# Full export
python tiles_exporter.py scan.ply out/similarity_transform.json out/tiles

# Quick preview: 10% of splats, SH degree 0
python tiles_exporter.py scan.ply out/similarity_transform.json out/preview \
    --fraction 0.1 --max-sh-degree 0
```

The same output works in both Cesium and ArcGIS Pro. ArcGIS Pro 3.7 recognizes it as a Gaussian Splat layer from `3DTILES_content_gltf` + `KHR_gaussian_splatting_compression_spz_2`; no Esri-specific extension is written.

---

## Step 3 — Check the tileset

`sanity_check.ipynb` loads a `tileset.json` and prints `extensionsUsed`, `extensionsRequired` and the `3DTILES_content_gltf` block.

Edit the path in the first cell to point at your output:

```python
d = json.load(open(r'out/tiles/tileset.json'))
```

What a correct splat tileset should show:

```
extensionsUsed: ['3DTILES_content_gltf']
extensionsRequired: ['3DTILES_content_gltf']
3DTILES_content_gltf: {'extensionsUsed': ['KHR_gaussian_splatting', 'KHR_gaussian_splatting_compression_spz_2'],
                       'extensionsRequired': ['KHR_gaussian_splatting', 'KHR_gaussian_splatting_compression_spz_2']}
```

---

## Step 4 (optional) — Package as `.3tz`

After Step 2, `out_dir` holds `tileset.json` plus one GLB per leaf tile, which can be hundreds of files. To keep things tidy for moving, sharing or archiving, pack the folder into a single `.3tz` with [3d-tiles-tools](https://github.com/CesiumGS/3d-tiles-tools):

```bash
npx 3d-tiles-tools convert -i out/tiles/tileset.json -o out/scan.3tz
```

`-i` also accepts the folder itself (`-i out/tiles`). `npx` downloads the tool on first run; Node.js is required.

> **Note:** this step is optional. The unpacked `tileset.json` folder already opens in both Cesium and ArcGIS Pro. `.3tz` only bundles everything into one file.

---

## Viewing the result

**CesiumJS**

Serve the output folder over HTTP (browsers block `file://`), then:

```js
const tileset = await Cesium.Cesium3DTileset.fromUrl("http://localhost:8080/tiles/tileset.json");
viewer.scene.primitives.add(tileset);
viewer.zoomTo(tileset);
```

Requires a CesiumJS release with `KHR_gaussian_splatting` support.

If you need to nudge the tileset vertically, apply the offset in an east-north-up frame (`Cesium.Transforms.eastNorthUpToFixedFrame`) rather than adding a raw `Cartesian3`, to avoid horizontal drift.

**Cesium ion**

Upload the output (zipped folder or `.3tz`) as 3D Tiles.

**ArcGIS Pro**

Open a **Local Scene** (not a 2D map) and add the tileset or `.3tz`.

---

## Output format details

**tileset.json**

- `asset.version = "1.1"`
- `extensionsUsed` / `extensionsRequired`: `3DTILES_content_gltf`
- `root.transform`: the 4×4 similarity matrix (column-major), local → ECEF
- Bounding volumes: oriented boxes in the local frame
- `geometricError`: bbox diagonal for internal nodes, `0.0` for leaves; `refine: "REPLACE"`

**tile_XXXX.glb**

- glTF 2.0 with one `POINTS` primitive
- `KHR_gaussian_splatting` + `KHR_gaussian_splatting_compression_spz_2` (required), `KHR_materials_unlit`
- The SPZ blob is stored in buffer view 0; accessors declare `POSITION`, `COLOR_0`, `KHR_gaussian_splatting:SCALE`, `KHR_gaussian_splatting:ROTATION` and SH coefficient attributes
- Node matrix converts the Z-up splat data to glTF's Y-up; 3D Tiles converts it back to Z-up at load time

**SPZ v3 (`spz_encode.py`)**

- Gzip-compressed, 16-byte header (magic `NGSP`, version 3, count, SH degree, fractional bits, flags)
- Positions as 24-bit fixed point (12 fractional bits)
- Alpha, color, log-scale as 8-bit; rotations packed as smallest-three quaternions (32 bits)
- SH: degree-1 at 5 bits, higher degrees at 4 bits (quantization bucket size)

Positions use 24-bit signed fixed point with 12 fractional bits, giving a range of about ±2048 PLY units at ~0.24 mm resolution. Tiles are not re-centered, so every splat must lie within ±2048 units of the PLY origin; values outside that range wrap silently. Keep the PLY in a local frame (as the pipeline expects) rather than projected or ECEF coordinates.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Only binary_little_endian PLY is supported.` | Re-export the PLY as binary little-endian |
| `Only float32 PLY properties are supported.` | PLY contains non-float properties (e.g. `uchar` colors); strip them |
| `SH degree: 0` despite `--max-sh-degree 3` | PLY has no / too few `f_rest_*` properties |
| Model floats above or sinks into terrain | Orthometric height passed as ellipsoidal (or vice versa); check `h = H + N`. Also note that global terrain basemaps are coarse. |
| Model is rotated or mirrored (manual anchor) | PLY axes are not X=East, Y=North, Z=Up; rotate the PLY first |
| Wrong position after rotating in SuperSplat | Use `--anchor 0,0,0`, not `bbox_center` |
| ArcGIS Pro: nothing shows | Opened in a 2D map; use a Local Scene |
| `Only N matched cameras ... (need ≥ 4)` | Camera names in the two inputs don't match |

---

## License

GPL-3.0. Portions derived from [dozeri83/geo-register-plugin](https://github.com/dozeri83/geo-register-plugin).
