# geomviewer

Quick viewer for vector datasets from the command line.

Supports:
- Shapefile (.shp)
- GeoJSON (.geojson, .json)
- GeoPackage (.gpkg)
- GeoParquet (.parquet, .geoparquet)

It automatically detects numeric columns and allows switching visualization columns.

---

## Installation
```bash
pip install viewgeom
```
> **Note:** Requires Python 3.9 or later.

---

## Usage
```bash
viewgeom <path> [--column <name>] [--layer <name>] [--limit N] [--simplify tol]
```

| Option                 | Description                                                     |
| ---------------------- | --------------------------------------------------------------- |
| `--column <name>`      | Choose numeric column for coloring                              |
| `--layer <name>`       | Select layer in a `.gpkg` file                                  |
| `--limit N`            | Max number of features to load (default: 100000)                |
| `--simplify <tol/off>` | Geometry simplification (default: `0.01`, use `off` to disable) |

### Examples
```bash
# View a GeoJSON
viewgeom gadm41_JPN_1.json

# Color by a numeric column
viewgeom landuse.shp --column area_sqkm

# View a GeoPackage and its specific layer
viewgeom countries.gpkg --layer ADM_ADM_2

# View a geoparquet
viewgeom mangrove_with_EAD.geoparquet --limit 150000 --simplify off
```
## Keyboard Controls
| Key        | Action                 |
| ---------- | ---------------------- |
| `+` / `-`  | Zoom in / out          |
| Arrow keys | Pan                    |
| `[` / `]`  | Switch numeric columns |
| `M`        | Switch colormap        |
| `B`        | Toggle basemap         |
| `R`        | Reset view             |

> **Notes**
>
> • For fast performance, only the first **100,000 features** are displayed by default. Adjust with `--limit` (e.g., `--limit 500000` or `--limit 0` for no limit).  
> • Complex geometries are simplified by default (`--simplify 0.01`).  
>   Use `--simplify off` to fully disable simplification.  
> • Basemap requires an active **internet connection**.  

## License
MIT License