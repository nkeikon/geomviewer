# DuckDB SQL Support (from v0.1.4)

`viewgeom` can filter and order attribute data using DuckDB before visualization. All queries run on a temporary in-memory table named `data`. Geometry is reattached after filtering or ordering so shapes can be drawn. 

Spatial queries are not supported yet. Aggregated results cannot be displayed because they cannot be matched back to geometry. Only a single table is available, so joins or multi-table queries are not supported.

## DuckDB Behavior in `viewgeom`
✔ What DuckDB expressions for `viewgeom` do
- Filter rows: any valid expression in WHERE is supported.
- Order rows: `ORDER BY` is applied before rendering or coloring.
- Create new columns: expressions in the `SELECT` list (with or without AS) become real columns in the output.
- Export computed columns to QGIS: when using `--qgis`, new columns created in the SQL query are saved to the GPKG and appear normally inside QGIS.

✘ What DuckDB expressions do not do in `viewgeom`
- It does not print tables or computed values: the viewer only prints basic info such as column lists and numeric ranges.
- It cannot display aggregated results: aggregations like `SUM` or `GROUP BY` work in DuckDB, but the results cannot be matched back to geometry, so they are not drawable.
- It does not save output files: DuckDB filters and computed columns run in memory only. To write the filtered results to a new file, use the separate `--save` option (e.g., `--save output.geojson`).

## Supported SQL features
### Column selection
- SELECT * FROM data
- SELECT name, pop FROM data

### WHERE filtering
- Comparisons: =, <>, <, >, <=, >=
- Logical operators: AND, OR, NOT
- Null checks: IS NULL, IS NOT NULL
- List tests: IN (...)

### String matching
- LOWER(), UPPER(), SUBSTR()
- LIKE, - ILIKE

### Numeric expressions
- area_ha / 100
- ROUND(pop / density, 2)
- CAST(col AS DOUBLE)

### Ordering and limiting
- ORDER BY name DESC
- LIMIT 50
- LIMIT 50 OFFSET 100

### Computed columns
- SELECT pop, pop / area AS density FROM data
- SELECT *, col*10 AS scaled FROM data

### Not supported
- Spatial SQL (ST_*)
- Multi-table queries or joins
- UNION, INTERSECT, EXCEPT
- Multiple dataset references
