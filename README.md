# viewgeom
[![Downloads](https://static.pepy.tech/badge/viewgeom)](https://pepy.tech/project/viewgeom)
[![PyPI version](https://img.shields.io/pypi/v/viewgeom)](https://pypi.org/project/viewgeom/)
[![Python version](https://img.shields.io/badge/python-%3E%3D3.9-blue.svg)](https://pypi.org/project/viewgeom/)

Quick viewer for vector datasets from the command line.

Supports:
- Shapefile (.shp)
- GeoJSON (.geojson, .json)
- GeoPackage (.gpkg)
- GeoParquet (.parquet, .geoparquet)

It automatically detects numeric columns and allows switching visualization columns.

## Installation
```bash
pip install viewgeom
```
> **Note:** Requires Python 3.9 or later.

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

## Credit & License
`viewgeom`, which followed from `viewtif`, was inspired by the NASA JPL Thermal Viewer — Semi-Automated Georeferencer (GeoViewer v1.12) developed by Jake Longenecker (University of Miami Rosenstiel School of Marine, Atmospheric & Earth Science) while at the NASA Jet Propulsion Laboratory, California Institute of Technology, with inspiration from JPL’s ECOSTRESS geolocation batch workflow by Andrew Alamillo. The original GeoViewer was released under the MIT License (2025) and may be freely adapted with citation.

## Citation
Longenecker, Jake; Lee, Christine; Hulley, Glynn; Cawse-Nicholson, Kerry; Purkis, Sam; Gleason, Art; Otis, Dan; Galdamez, Ileana; Meiseles, Jacquelyn. GeoViewer v1.12: NASA JPL Thermal Viewer—Semi-Automated Georeferencer User Guide & Reference Manual. Jet Propulsion Laboratory, California Institute of Technology, 2025. PDF.

## License
This project is released under the MIT License © 2025 Keiko Nomura.

If you find this tool useful, please consider supporting or acknowledging the author. 
