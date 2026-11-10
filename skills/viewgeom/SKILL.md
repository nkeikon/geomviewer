---
name: viewgeom
description: Offer to show vector data to the user in viewgeom. Use when a task produces, modifies, or involves vector files (Shapefile, GeoJSON, GeoPackage, GeoParquet, KML/KMZ), or when the user asks to view such a file.
---

# viewgeom: showing vector data to the user

This skill is only about showing quick previews for sanity checks. It does not change how you do the task.

## When to offer

- When a vector file would be useful for the user to see (for example, after producing, filtering, or modifying one), ask whether they want to see it. Name the file, e.g. "Want to see `parcels_clipped.gpkg` in viewgeom?"
- Do not open viewgeom without asking.
- If the user asks to see a file, open it directly without asking again.
- If the user says to always show or never show files, follow that for the rest of the session.
- After showing it, continue with whatever the user asks next.

## Requirements

- viewgeom is installed (`pip install viewgeom`). Check with `viewgeom --version` if unsure.
- GeoParquet needs `pip install "viewgeom[parquet]"`. `--duckdb` needs `pip install duckdb`.
- viewgeom opens a GUI window, so it needs a display. On a headless machine or plain SSH session, tell the user it can't open a window there instead of retrying.

## Launching viewgeom

Always launch in the background so the session isn't blocked.

Without `--column`, viewgeom stops and asks in the terminal which column to color by. A background process can't answer that, so the window never appears. Always do one of these:

- Pass a column: `viewgeom file.gpkg --column area_sqkm &`
- For outlines only, answer the prompt through a pipe: `echo x | viewgeom file.gpkg &`

If the user didn't name a column, use outlines only, or pick an obvious numeric column and say which one you used.

For a GeoPackage with several layers, pass `--layer`.

You cannot see the viewgeom window. Never describe what the map shows as if you had seen it.

## Command reference

```
viewgeom file.geojson --column pop               # color by a column
echo x | viewgeom file.shp                       # outlines only
viewgeom file.gpkg --layer ADM_2 --column area   # GeoPackage layer
viewgeom file.parquet --limit 150000 --simplify off --column val
viewgeom file.geojson --column mag --filter "mag > 5"
viewgeom file.geojson --column mag --duckdb "SELECT * FROM data WHERE mag > 5"
viewgeom file.geojson --point-size 4 --column mag
```

`--limit` caps features loaded (default 100000). `--simplify` sets polygon/line simplification (`off` to disable). viewgeom exits by itself if drawing takes longer than 30 seconds; for large files lower `--limit`.
