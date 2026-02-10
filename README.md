# viewgeom
[![Downloads](https://static.pepy.tech/badge/viewgeom)](https://pepy.tech/project/viewgeom)
[![PyPI version](https://img.shields.io/pypi/v/viewgeom)](https://pypi.org/project/viewgeom/)
[![Python version](https://img.shields.io/badge/python-%3E%3D3.9-blue.svg)](https://pypi.org/project/viewgeom/)

Quick viewer for vector datasets from the command line.

The viewer now includes data filtering and processing tools (>v0.1.4).
You can run SQL queries with [DuckDB](duckdb.md) for fast and flexible selection or apply attribute filters with pandas query syntax. You can also send the dataset to QGIS with one command for full desktop visualization, or save as a new file with `--save`.

Supports:
- Shapefile (.shp)
- GeoJSON (.geojson, .json)
- GeoPackage (.gpkg)
- GeoParquet (.parquet, .geoparquet)
- FileGDB (.gdb)*
- KML, KMZ  (.kml, .kmz)*

It automatically detects layers and columns and allows switching visualization columns.

**Saving to KML, KMZ, or FileGDB is not supported. Please use GeoPackage, GeoJSON, JSON, Shapefile, or Parquet for full attribute export (see "Update in v0.1.4").*
## Need help?

You can ask questions about usage via the documentation-based chatbot:  
👉 [Ask the viewtif + viewgeom + viewinline Helper](https://chatgpt.com/g/g-698b61c42f788191b884aed1b99dfcd8-viewtif-viewgeom-viewinline-helper)

## Installation
```bash
pip install viewgeom
```
> **Note:** Requires Python 3.9 or later.

GeoParquet and Parquet support requires pyarrow, which is optional:
```bash
pip install "viewgeom[parquet]"
```

DuckDB support requires duckdb:
```bash
pip install duckdb
```

## Usage
```bash
viewgeom <path> [--column <name>] [--layer <name>] \
         [--filter <expr>] [--duckdb <SQL>] \
         [--limit <N>] [--simplify <tol|off>] \
         [--point-size <px>] [--qgis] [--save <file>]
```

## Options

| Option                 | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `--column <name>`      | Choose a numeric or categorical column for coloring                         |
| `--layer <name>`       | Select a layer in a `.gpkg` file                                            |
| `--limit N`            | Max number of features to load (default: 100000)                            |
| `--simplify <tol/off>` | Geometry simplification; numeric tolerance or `off` to disable              |
| `--point-size px`      | Set point size in pixels (overrides automatic sizing)                       |
| `--filter "<expr>"`    | Filter using pandas query syntax                                            |
| `--duckdb "<SQL>"`     | Attribute-only SQL filtering using DuckDB                                   |
| `--qgis`               | Export results to a temporary GeoPackage and open in QGIS                   |
| `--save <path>`        | Save filtered results to the provided file path; format is determined by the filename (e.g. filtered.json, result.kmz, subset.gpkg)                |
| `--version`            | Show version information                                                    |


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

# Attribute filtering with pandas
viewgeom earthquake.geojson --filter "mag > 5"

# Attribute filtering with DuckDB and export to QGIS
viewgeom earthquake.geojson --duckdb "SELECT * FROM data WHERE mag > 5" --qgis

# Save the output as a new file
viewgeom earthquake.geojson --duckdb "SELECT mag FROM data WHERE mag > 5" --save "filtered.geojson"
```
## Keyboard Controls
| Key        | Action                 |
| ---------- | ---------------------- |
| `+` / `-`  | Zoom in / out          |
| Arrow keys | Pan                    |
| `[` / `]`  | Switch columns         |
| `M`        | Switch colormap        |
| `B`        | Toggle basemap         |
| `R`        | Reset view             |

> **Notes**
>
> • For fast performance, only the first **100,000 features** are displayed by default. Adjust with `--limit` (e.g., `--limit 500000` or `--limit 0` for no limit).  
> • Complex geometries are simplified by default (`--simplify 0.01`).  
>   Use `--simplify off` to fully disable simplification.  

### Update in v0.1.3
- `Viewgeom` supports both numeric and categorical columns for visualization, while still giving the option to display outlines only by entering 'x' when prompted.
- For a numeric column, the tool prints the data range as well as the visualization stretch range. For a categorical column, it prints the number of unique categories and show the first few (up to five).
- For point data, the tool prints the automatically chosen point size. You can override this by using the `--point-size` option.
- KML/KMZ files are fully supported and their attribute columns can be used for coloring.
- If an internet connection is slow or unavailable, the basemap will be sipped and the viewer will continue without it.
- The default behavior still limits large datasets to 100000 features. In addition to this, if the feature density is high, the tool further limits the sample to 1000 features. Users can adjust this behavior with the `--limit` option.
- As a safeguard, if drawing takes more than 30 seconds, viewgeom will exit automatically.

### Update in v0.1.4
- DuckDB SQL support is now available with `--duckdb`, allowing you to query any vector dataset using SQL. It works with .shp, .geojson, .gpkg, .parquet, .geoparquet, .kml, and .kmz, and supports column selection, filtering, ordering, and numeric expressions (current limitations: only one dataset at a time, and no spatial queries yet). For more information, please see [DuckDB SQL Support](duckdb.md).
- The `--filter` option evaluates expressions using pandas query syntax. It is a lightweight alternative to SQL for simple filtering.
- The `--qgis` export option lets you send filtered results directly to QGIS. A temporary .gpkg file is generated, and QGIS opens automatically on macOS, Linux, and Windows.
- FileGDB (.gdb) is now supported. Use `--layer` to select one if it contains multiple layers.
- You can save outputs with `--save`. The file format is determined by the filename you provide (e.g., `--save filtered.json`, `--save result.shp`, or `--save subset.gpkg`). Saving to KML or KMZ is disabled because the OGR KML driver (used through pyogrio) does not preserve attribute fields. FileGDB export is also disabled, since full write support requires the proprietary ESRI FileGDB driver. Please use GeoPackage, GeoJSON, JSON, Shapefile, or Parquet for complete attribute export.

## Credit & License
`viewgeom`, which followed from `viewtif`, was inspired by the NASA JPL Thermal Viewer — Semi-Automated Georeferencer (GeoViewer v1.12) developed by Jake Longenecker (University of Miami Rosenstiel School of Marine, Atmospheric & Earth Science) while at the NASA Jet Propulsion Laboratory, California Institute of Technology, with inspiration from JPL’s ECOSTRESS geolocation batch workflow by Andrew Alamillo. The original GeoViewer was released under the MIT License (2025) and may be freely adapted with citation.

## Citation
Longenecker, Jake; Lee, Christine; Hulley, Glynn; Cawse-Nicholson, Kerry; Gleason, Art; Otis, Dan; Galdamez, Ileana; Meiseles, Jacquelyn. GeoViewer v1.12: NASA JPL Thermal Viewer—Semi-Automated Georeferencer User Guide & Reference Manual. Jet Propulsion Laboratory, California Institute of Technology, 2025. PDF.

## License
This project is released under the MIT License © 2025 Keiko Nomura.

If you find this tool useful, please consider supporting or acknowledging the author. 

## Useful links
- [YouTube demo playlist](https://www.youtube.com/playlist?list=PLP9MNCMgJIHj6FvahJ6Tembp1rCyhLtR4)
- [Featured by Matt Forrest!](https://www.linkedin.com/posts/mbforr_sometimes-to-make-big-leaps-forward-we-have-activity-7391837368929955840-s0O0?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAA0INsBVIO1f6nS_NkKqFh4Na1ZpoYo2fc)
- [Demo at the initial release](https://www.linkedin.com/posts/keiko-nomura-0231891_dont-you-sometimes-just-want-to-see-a-activity-7388654251562102784-L_iX?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAA0INsBVIO1f6nS_NkKqFh4Na1ZpoYo2fc)
- [Demo for v0.1.3 release](https://www.linkedin.com/posts/keiko-nomura-0231891_i-just-released-viewgeom-v013-it-has-more-activity-7397803657238347776--aiT?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAA0INsBVIO1f6nS_NkKqFh4Na1ZpoYo2fc)
