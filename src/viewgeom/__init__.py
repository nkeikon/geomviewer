#!/usr/bin/env python3
"""
viewgeom — Interactive viewer for vector datasets (.shp, .geojson, .gpkg, .parquet, .geoparquet, .kml, .kmz)
"""

import sys, os, zipfile, tempfile
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.ops import unary_union
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QGraphicsPathItem, QGraphicsEllipseItem, QStatusBar, QScrollBar
)
from PySide6.QtGui import QPen, QColor, QPainterPath, QPainter
from PySide6.QtCore import Qt

__version__ = "0.1.3"

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def flatten_geometry(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "GeometryCollection":
        parts = [g for g in geom.geoms if not g.is_empty]
        return unary_union(parts) if parts else None
    return geom


def load_vector_any(path, layer=None, limit=100_000, simplify=0.01):
    ext = os.path.splitext(path)[1].lower()

    # --- GeoParquet support ---
    if ext in (".parquet", ".geoparquet"):
        try:
            import pyarrow
        except Exception:
            raise ImportError(
                "Reading Parquet requires the optional dependency 'pyarrow'.\n"
                "Install it with:\n"
                "    pip install 'viewgeom[parquet]'"
            )

        gdf = gpd.read_parquet(path)

    # --- Shapefile / GeoJSON / JSON ---
    elif ext in (".shp", ".geojson", ".json"):
        gdf = gpd.read_file(path)

    # --- GeoPackage ---
    elif ext == ".gpkg":
        try:
            import pyogrio
            available_layers = [lyr[0] for lyr in pyogrio.list_layers(path)]

            if layer:
                if layer not in available_layers:
                    raise ValueError(f"Layer '{layer}' not found. Available: {available_layers}")
                gdf = gpd.read_file(path, layer=layer)
                print(f"[INFO] Loaded GPKG layer: {layer}")

            else:
                default_layer = available_layers[0]
                gdf = gpd.read_file(path, layer=default_layer)
                print(f"[INFO] Loaded GPKG layer: {default_layer}")
                layer = default_layer

                if len(available_layers) > 1:
                    print("[INFO] Other layers available:")
                    for name in available_layers[1:]:
                        print(f"   • {name}")
                    print("Use: --layer <name> to load a different one")

        except ImportError:
            print("[INFO] pyogrio not installed — loading default layer only.")
            gdf = gpd.read_file(path)

    # --- KML / KMZ ---
    elif ext in (".kml", ".kmz"):
        try:
            # Extract or read the KML
            if ext == ".kml":
                gdf = gpd.read_file(path, driver="KML")

            else:
                # KMZ: unzip and read the KML inside
                with zipfile.ZipFile(path, "r") as zf:
                    kml_files = [n for n in zf.namelist() if n.lower().endswith(".kml")]
                    if not kml_files:
                        raise ValueError("No .kml file found inside KMZ archive.")
                    kml_name = kml_files[0]

                    tmpdir = tempfile.mkdtemp(prefix="viewgeom_kmz_")
                    extracted_path = zf.extract(kml_name, tmpdir)
                    kml_path = os.path.abspath(extracted_path)

                gdf = gpd.read_file(kml_path, driver="KML")

            # Keep valid geometries only
            gdf = gdf[gdf.geometry.notnull() & gdf.geometry.is_valid]

        except Exception as e:
            raise RuntimeError(f"Failed to load KML/KMZ file: {e}")

    # --- CRS handling ---
    if gdf.crs is None:
        print("[WARN] No CRS found — assuming EPSG:4326")
        gdf.set_crs(4326, inplace=True)
    else:
        print(f"[INFO] CRS detected: {gdf.crs.to_string()}")

    # --- Fix GeometryCollections ---
    if "GeometryCollection" in gdf.geom_type.unique():
        print("[INFO] Flattening GeometryCollections")
        gdf["geometry"] = gdf.geometry.apply(flatten_geometry)
        gdf = gdf[gdf.geometry.notnull()]

    # --- Limit features for very large or very dense sets ---
    n = len(gdf)
    large_threshold = 100_000

    minx, miny, maxx, maxy = gdf.total_bounds
    area = max((maxx - minx) * (maxy - miny), 1e-12)
    density = n / area

    dense_threshold = 300_000
    dense_limit = 1_000

    # 1. Automatic density-based sampling
    if density > dense_threshold:
        print(
            f"[WARN] Extremely dense dataset: n={n:,}, area={area:.6f} deg² \n"
            f"[INFO] Sampling down to {dense_limit:,} features" #, density={density:,.0f} features/unit²)"
        )
        # print(f"[WARN] Density-based sampling → {dense_limit:,} features")
        gdf = gdf.sample(dense_limit, random_state=42)
        n = len(gdf)

    # 2. Automatic large-dataset rule
    if n > large_threshold:
        print(f"[WARN] Large dataset ({n:,} features) — sampling {limit:,}")
        gdf = gdf.sample(limit, random_state=42)
        n = len(gdf)

    # 3. Final user limit fallback (for cases where limit < large_threshold)
    if n > limit:
        print(f"[INFO] User-specified limit → sampling down to {limit:,} features")
        gdf = gdf.sample(limit, random_state=42)

    # --- Simplify large, complex polygons ---
    geom_type = gdf.geom_type.mode()[0]
    if geom_type not in ("Point", "MultiPoint"):
        vertex_counts = []
        for g in gdf.geometry.head(500):
            if g is None or g.is_empty:
                continue
            try:
                if g.geom_type == "Polygon":
                    vertex_counts.append(len(g.exterior.coords))
                elif g.geom_type == "MultiPolygon":
                    vertex_counts.extend(len(p.exterior.coords) for p in g.geoms)
                elif g.geom_type == "LineString":
                    vertex_counts.append(len(g.coords))
                elif g.geom_type == "MultiLineString":
                    vertex_counts.extend(len(l.coords) for l in g.geoms)
            except Exception:
                continue

        avg_vertices = np.mean(vertex_counts) if vertex_counts else 0
        if len(gdf) > 5000 or (avg_vertices > 2000 and len(gdf) > 200):
            print(f"[INFO] Simplifying geometries (tol={simplify})")
            try:
                gdf["geometry"] = gdf.geometry.simplify(simplify, preserve_topology=True)
            except Exception as e:
                print(f"[WARN] Simplify failed — continuing without simplification ({e})")

    # --- Remove invalid empties ---
    gdf = gdf[~gdf.geometry.is_empty]

    return gdf

# ---------------------------------------------------------------------
# Color mapping
# ---------------------------------------------------------------------
def get_color_mapping(gdf, column, cmap_name="viridis"):
    # If no column selected → no colors
    if column is None:
        return [None] * len(gdf), None
    
    series = gdf[column].dropna()

    if pd.api.types.is_numeric_dtype(series):
        pmin, pmax = np.percentile(series, [5, 95])
        if abs(pmax - pmin) < 1e-9:
            vmin, vmax = series.min(), series.max()
            print(f"[WARN] Low variance — using full range ({vmin}–{vmax})")
        else:
            vmin, vmax = pmin, pmax
        cmap = plt.get_cmap(cmap_name)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        colors = series.map(lambda x: cmap(norm(x)))
        
        return colors, (vmin, vmax, pmin, pmax)

    # Fallback for columns that are neither numeric nor categorical:
    colors = [None] * len(gdf)
    return colors, None

def is_categorical(series):
    # Explicit pandas CategoricalDtype
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True

    # Strings / object columns are categorical
    if pd.api.types.is_object_dtype(series):
        return True

    # DO NOT treat integer/float as categorical
    if pd.api.types.is_numeric_dtype(series):
        return False

    # Fallback: treat booleans as categorical
    if pd.api.types.is_bool_dtype(series):
        return True

    return False

def categorical_colors(series):
    unique_vals = sorted(series.dropna().unique())
    cmap = plt.get_cmap("tab20", len(unique_vals))

    color_map = {
        val: cmap(i) for i, val in enumerate(unique_vals)
    }
    return color_map

# ---------------------------------------------------------------------
# Graphics view (zoom/pan)
# ---------------------------------------------------------------------
class VectorView(QGraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._wheel_zoom_step = 1.2

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = self._wheel_zoom_step if delta > 0 else 1 / self._wheel_zoom_step
        self.scale(factor, factor)

# ---------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------
class VectorViewer(QMainWindow):
    def __init__(self, path, column=None, limit=100_000, simplify=0.01, layer=None, point_size=None):
        super().__init__()
        # print(f"[INFO] Loading {path}")
        self.path = path
        self.layer = layer
        self.user_point_size = point_size  

        # --- Handle simplify argument ---
        if isinstance(simplify, str) and simplify.lower() == "off":
            simplify = 0.0  # fully disable simplification
        else:
            try:
                simplify = float(simplify)
            except ValueError:
                print(f"[WARN] Invalid simplify '{simplify}', using default 0.01")
                simplify = 0.01

        self.simplify = simplify

        # Load vector *after* simplify normalization
        self.gdf = load_vector_any(path, layer, limit, self.simplify)

        minx, miny, maxx, maxy = self.gdf.total_bounds

        if self.user_point_size is not None:
            # explicit override from CLI
            self.point_size = float(self.user_point_size)
        else:
            # the default setting
            extent = max(maxx - minx, maxy - miny)
            self.point_size = max(1.5, min(6, extent * 0.001))

        # Only show point size when dataset actually contains points
        geom_types = set(self.gdf.geom_type.unique())
        if "Point" in geom_types or "MultiPoint" in geom_types:
            print(f"[INFO] Point size: {self.point_size}")

        self.colormaps = [
            "plasma",    # default continuous
            "turbo",     # bold, web-mapping look
            "cividis",   # accessible & balanced
            "Spectral",  # diverging / strong variation
            "tab10"      # categorical
        ]

        self.cmap_index = 0

        # Different colormaps for categorical columns
        self.cat_colormaps = ["tab20", "Set3", "Accent"]
        self.cat_cmap_index = 0

        self.scene = QGraphicsScene(self)
        self.view = VectorView(self.scene)
        self.setCentralWidget(self.view)
        self.setStatusBar(QStatusBar())
        self.basemap_items = []
        self.feature_items = []

        # Basemap will be loaded later depending on numeric columns
        self.base_gdf = None

        # ---- Skip numeric column selection for line geometries ----
        if all(gt in ("LineString", "MultiLineString") for gt in self.gdf.geom_type.unique()):
            self.color_col = None
            print("[INFO] Outlines only")
        else:
            # ---- Detect numeric and categorical columns ----
            num_cols = [c for c in self.gdf.columns if self.gdf[c].dtype.kind in "if"]
            cat_cols = [c for c in self.gdf.columns if is_categorical(self.gdf[c])]

            self.num_cols = num_cols
            self.cat_cols = cat_cols

            # unified ordered list
            all_cols = num_cols + cat_cols
            self.all_cols = all_cols

            self._has_numeric_cols = len(num_cols) > 0

            # If user specified a column and it exists
            if column and (column in num_cols or column in cat_cols):
                self.color_col = column
                print(f"[INFO] Coloring by: {self.color_col}")

            else:
                print("[INFO] Available columns:")
                for i, c in enumerate(all_cols):
                    dtype = "numeric" if c in num_cols else "categorical"
                    print(f"  [{i}] {c}   ({dtype})")

                choice = input("Select column index or 'x' for outlines only: ").strip().lower()

                if choice == "x":
                    print("[INFO] Outlines only.")
                    self.color_col = None

                else:
                    try:
                        idx = int(choice)
                        if idx < len(all_cols):
                            self.color_col = all_cols[idx]
                            print(f"[INFO] Coloring by: {self.color_col}")
                        else:
                            print("[WARN] Invalid index — outlines only.")
                            self.color_col = None

                    except Exception:
                        print("[WARN] Invalid selection — outlines only.")
                        self.color_col = None

        # ---- Basemap behavior ----
        if getattr(self, "_has_numeric_cols", False):
            if self.base_gdf is None:
                self._load_basemap()
            if self.base_gdf is not None:
                self._draw_basemap()
                print("[INFO] Basemap displayed (Press 'B' to remove)")
        else:
            print("[INFO] Basemap optional (Press 'B' to add)")

        self._update_window_title()

        # ---- Color mapping ----
        if self.color_col is None:
            self.gdf["_color"] = None
            stats = None

        elif self.color_col in num_cols:
            # numeric coloring (existing)
            self.gdf["_color"], stats = get_color_mapping(self.gdf, self.color_col)

        elif self.color_col in self.cat_cols:
            print(f"[INFO] Using categorical coloring for '{self.color_col}'")

            series = self.gdf[self.color_col]
            uniques = sorted(series.dropna().unique(), key=lambda x: str(x))
            n_unique = len(uniques)

            if n_unique <= 5:
                print(f"[INFO] Categories: {n_unique} total")
                for val in uniques:
                    print(f"   • {val!r}")
            else:
                print(f"[INFO] Categories: {n_unique} total")
                print(f"[INFO] Showing first 5 categories:")
                for val in uniques[:5]:
                    print(f"   • {val!r}")

            # Assign colors
            cmap_dict = categorical_colors(self.gdf[self.color_col])
            self.category_color_map = cmap_dict

            # convert to RGBA tuples
            self.gdf["_color"] = self.gdf[self.color_col].map(
                lambda v: cmap_dict.get(v, (0.7, 0.7, 0.7, 1))
            )
            stats = None

        if stats:
            vmin, vmax, _, _ = stats

            col = self.gdf[self.color_col]

            # Valid/missing counts
            valid_count = col.count()
            missing_count = col.isna().sum()

            # Dataset min/max (true)
            data_min = col.min(skipna=True)
            data_max = col.max(skipna=True)

            print(f"[INFO] Valid values: {valid_count:,}  Missing: {missing_count:,}")
            print(f"[INFO] Dataset min/max (non-NaN): {data_min:.3f} to {data_max:.3f}")
            print(f"[INFO] Stretch range (p5–p95, non-NaN): {vmin:.3f} to {vmax:.3f}")

        # ---- Draw features ----
        self._draw_geoms()

        # ---- Scene extents ----
        minx, miny, maxx, maxy = self.gdf.total_bounds
        self.scene.setSceneRect(minx, -maxy, maxx - minx, maxy - miny)
        # self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.resize(1000, 800)
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        QApplication.processEvents()
        self.initial_transform = self.view.transform()
        
        print(f"[INFO] Features displayed: {len(self.gdf):,}")

    # -----------------------------------------------------------------
    def _load_basemap(self):
        import requests
        from io import BytesIO

        url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"

        print("[INFO] Loading basemap (timeout 3s)...")

        # --- Step 1. Download ONLY (timeout enforced) ---
        try:
            resp = requests.get(url, timeout=3)
            resp.raise_for_status()
        except Exception:
            print("[WARN] Basemap not loaded (download failed or too slow).")
            self.base_gdf = None
            return

        # --- Step 2. If download succeeded, then parse ---
        try:
            zip_bytes = BytesIO(resp.content)
            gdf = gpd.read_file(zip_bytes)

            if gdf.crs != self.gdf.crs:
                gdf = gdf.to_crs(self.gdf.crs)

            self.base_gdf = gdf

        except Exception as e:
            print(f"[WARN] Basemap not loaded (read or CRS error: {e})")
            self.base_gdf = None
            return

    def _draw_basemap(self):
        if self.base_gdf is None:
            return

        palette = QApplication.palette()
        bg = palette.window().color()
        brightness = (bg.red() * 299 + bg.green() * 587 + bg.blue() * 114) / 1000
        pen = QPen(QColor(255, 255, 255) if brightness < 128 else QColor(80, 80, 80))
        pen.setWidthF(0.5)
        pen.setCosmetic(True)

        for it in self.basemap_items:
            self.scene.removeItem(it)
        self.basemap_items.clear()

        for geom in self.base_gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            path = QPainterPath()
            geoms = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
            for g in geoms:
                if g.geom_type == "Polygon":
                    try:
                        for i, (x, y) in enumerate(g.exterior.coords):
                            y = -y
                            path.moveTo(x, y) if i == 0 else path.lineTo(x, y)
                        path.closeSubpath()
                    except Exception:
                        continue
                elif g.geom_type == "LineString":
                    try:
                        for i, (x, y) in enumerate(g.coords):
                            y = -y
                            path.moveTo(x, y) if i == 0 else path.lineTo(x, y)
                    except Exception:
                        continue
            item = QGraphicsPathItem(path)
            item.setPen(pen)
            item.setZValue(-100)
            self.scene.addItem(item)
            self.basemap_items.append(item)

    def _color_for_value(self, val):
        if isinstance(val, (tuple, list, np.ndarray)):
            r, g, b, a = [int(255 * v) for v in val]
            return QColor(r, g, b, a)
        elif isinstance(val, str):
            return QColor(val)
        else:
            # fallback for no numeric column: use edge-only color
            palette = QApplication.palette()
            bg = palette.window().color()
            brightness = (bg.red() * 299 + bg.green() * 587 + bg.blue() * 114) / 1000
            return QColor(220, 220, 220) if brightness < 128 else QColor(60, 60, 60)

    def _draw_geoms(self):
        
        # Remove existing feature items (if tracked)
        for it in getattr(self, "feature_items", []):
            self.scene.removeItem(it)
        self.feature_items = []

        pen = QPen()
        pen.setWidthF(0.8)
        pen.setCosmetic(True)

        # --- Freeze protection timer ---
        from PySide6.QtCore import QElapsedTimer
        timer = QElapsedTimer()
        timer.start()

        for _, row in self.gdf.iterrows():

            # Abort if drawing takes too long
            if timer.elapsed() > 30000:  # 30 seconds
                print("[WARN] Drawing aborted (took too long)")
                return

            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            
            # Choose color and brush per feature
            palette = QApplication.palette()
            bg = palette.window().color()
            brightness = (bg.red() * 299 + bg.green() * 587 + bg.blue() * 114) / 1000

            # Decide brush and pen based on column type
            if self.color_col in getattr(self, "num_cols", []):
                # numeric
                if row["_color"] is None:
                    brush = Qt.NoBrush
                else:
                    color = self._color_for_value(row["_color"])
                    brush = color

                # thin contrasting edge
                edge_color = QColor(255, 255, 255) if brightness < 128 else QColor(0, 0, 0)
                pen.setColor(edge_color)
                pen.setWidthF(0.3)

            elif self.color_col in getattr(self, "cat_cols", []):
                # categorical
                rgba = row["_color"]
                if rgba is None:
                    brush = Qt.NoBrush
                    pen.setColor(QColor(120, 120, 120))
                    pen.setWidthF(0.6)
                else:
                    r, g, b, a = [int(255 * c) for c in rgba]
                    color = QColor(r, g, b, a)
                    brush = color

                    pen.setColor(QColor(40, 40, 40))
                    pen.setWidthF(0.3)

            else:
                # outlines only mode
                brush = Qt.NoBrush
                if brightness < 128:
                    color = QColor(255, 80, 80)
                else:
                    color = QColor(150, 0, 0)

                pen.setColor(color)
                pen.setWidthF(0.8)


            geoms = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
            for g in geoms:
                path = QPainterPath()

                if g.geom_type == "Polygon":
                    try:
                        for i, (x, y) in enumerate(g.exterior.coords):
                            y = -y
                            path.moveTo(x, y) if i == 0 else path.lineTo(x, y)
                        path.closeSubpath()
                        for ring in g.interiors:
                            for i, (x, y) in enumerate(ring.coords):
                                y = -y
                                path.moveTo(x, y) if i == 0 else path.lineTo(x, y)
                            path.closeSubpath()
                    except Exception:
                        continue

                elif g.geom_type == "LineString":
                    try:
                        for i, (x, y) in enumerate(g.coords):
                            y = -y
                            path.moveTo(x, y) if i == 0 else path.lineTo(x, y)
                    except Exception:
                        continue

                elif g.geom_type == "Point":
                    s = self.point_size
                    item = QGraphicsEllipseItem(g.x - s/2, -g.y - s/2, s, s)
                    item.setPen(pen)
                    item.setBrush(brush if self.color_col else pen.color())
                    self.scene.addItem(item)
                    self.feature_items.append(item)
                    continue

                item = QGraphicsPathItem(path)
                item.setPen(pen)
                item.setBrush(brush)
                self.scene.addItem(item)
                self.feature_items.append(item)

    def _update_window_title(self):
        filename = os.path.basename(self.path)
        parts = []

        if self.layer:
            parts.append(self.layer)

        if self.color_col:
            parts.append(self.color_col)

        parts.append(filename)

        self.setWindowTitle(" — ".join(parts))

    def _switch_column(self, direction):
        # Use the original unified ordering
        all_cols = self.all_cols

        if not all_cols:
            print("[INFO] No columns available to switch.")
            return

        # find current index or default to first
        try:
            idx = all_cols.index(self.color_col)
        except ValueError:
            idx = 0

        # wrap around
        idx = (idx + direction) % len(all_cols)
        self.color_col = all_cols[idx]
        print(f"[INFO] Coloring by: {self.color_col}")
        self._update_window_title()

        # numeric branch
        if self.color_col in self.num_cols:
            cmap_name = self.colormaps[self.cmap_index]
            self.gdf["_color"], stats = get_color_mapping(
                self.gdf, self.color_col, cmap_name=cmap_name
            )

            if stats:
                vmin, vmax, pmin, pmax = stats
                print(f"[INFO] Value range: min={vmin:.3f}, max={vmax:.3f}")
                print(f"[INFO] Percentile stretch: p5={pmin:.3f}, p95={pmax:.3f}")

        # categorical branch
        else:
            series = self.gdf[self.color_col]
            uniques = sorted(series.dropna().unique(), key=lambda x: str(x))
            n_unique = len(uniques)

            # Show category info (same style as startup)
            if n_unique <= 5:
                print(f"[INFO] Categories: {n_unique} total")
                for val in uniques:
                    print(f"   • {val!r}")
            else:
                print(f"[INFO] Categories: {n_unique} total")
                print("[INFO] Showing first 5 categories:")
                for val in uniques[:5]:
                    print(f"   • {val!r}")

            # Color assignment
            cmap_name = self.cat_colormaps[self.cat_cmap_index]
            cmap = plt.get_cmap(cmap_name)

            color_map = {u: cmap(i / max(1, n_unique - 1)) for i, u in enumerate(uniques)}

            self.gdf["_color"] = self.gdf[self.color_col].map(
                lambda v: color_map.get(v, (0.7, 0.7, 0.7, 1))
            )

        self._draw_geoms()
        if self.basemap_items:
            self._draw_basemap()

    # -----------------------------------------------------------------
    def keyPressEvent(self, ev):
        k = ev.key()
        hsb, vsb = self.view.horizontalScrollBar(), self.view.verticalScrollBar()
        if k in (Qt.Key.Key_Plus, Qt.Key.Key_Equal, Qt.Key.Key_Z):
            self.view.scale(1.2, 1.2)
        elif k in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore, Qt.Key.Key_X):
            self.view.scale(1/1.2, 1/1.2)
        elif k in (Qt.Key.Key_Left, Qt.Key.Key_A):
            hsb.setValue(hsb.value() - 50)
        elif k in (Qt.Key.Key_Right, Qt.Key.Key_D):
            hsb.setValue(hsb.value() + 50)
        elif k in (Qt.Key.Key_Up, Qt.Key.Key_W):
            vsb.setValue(vsb.value() - 50)
        elif k in (Qt.Key.Key_Down, Qt.Key.Key_S):
            vsb.setValue(vsb.value() + 50)
        elif k == Qt.Key.Key_M:
            if not self.color_col:
                print("[INFO] Colormap switching disabled (no color-by column).")
                return

            # numeric column
            if self.color_col in self.num_cols:
                self.cmap_index = (self.cmap_index + 1) % len(self.colormaps)
                cmap_name = self.colormaps[self.cmap_index]
                print(f"[INFO] Switched numeric colormap to: {cmap_name}")
                self.gdf["_color"], _ = get_color_mapping(self.gdf, self.color_col, cmap_name=cmap_name)

            # categorical column
            elif self.color_col in self.cat_cols:
                self.cat_cmap_index = (self.cat_cmap_index + 1) % len(self.cat_colormaps)
                cmap_name = self.cat_colormaps[self.cat_cmap_index]
                print(f"[INFO] Switched categorical colormap to: {cmap_name}")

                cmap = plt.get_cmap(cmap_name)
                uniques = sorted(self.gdf[self.color_col].dropna().unique())

                color_map = {u: cmap(i / max(1, len(uniques)-1)) for i, u in enumerate(uniques)}

                # update colors
                self.gdf["_color"] = self.gdf[self.color_col].map(
                    lambda v: color_map.get(v, (0.7, 0.7, 0.7, 1))
                )

            self._draw_geoms()

        elif k in (Qt.Key.Key_BraceRight, Qt.Key.Key_BracketRight):  # ]
            if self.color_col:
                self._switch_column(+1)
        elif k in (Qt.Key.Key_BraceLeft, Qt.Key.Key_BracketLeft):  # [
            if self.color_col:
                self._switch_column(-1)
        elif k == Qt.Key.Key_B:
            if self.basemap_items:
                # Basemap currently visible as default
                for it in self.basemap_items:
                    self.scene.removeItem(it)
                self.basemap_items.clear()
                print("[INFO] Basemap removed")
            else:
                if self.base_gdf is None:
                    self._load_basemap()

                if self.base_gdf is not None:
                    self._draw_basemap()
                    print("[INFO] Basemap displayed")

        elif k == Qt.Key.Key_R:
            self.view.resetTransform()
            self.view.setTransform(self.initial_transform)
            print("[INFO] Reset view")
        else:
            super().keyPressEvent(ev)

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "Quick view for vector datasets (.shp, .geojson, .gpkg, .parquet, .geoparquet)\n\n"
            "Controls:\n"
            "  + / -  : zoom in/out\n"
            "  arrows : pan\n"
            "  [ / ]  : switch numeric columns (if available)\n"
            "  M      : switch colormap (numeric only)\n"
            "  B      : toggle basemap\n"
            "  R      : reset view"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
    "--version",
    action="version",
    version=f"viewgeom {__version__}"
    )
    parser.add_argument(
        "path",
        help="Path to vector file"
    )
    parser.add_argument(
        "--column",
        type=str,
        help="Numeric column name to color by"
    )
    parser.add_argument(
        "--layer",
        type=str,
        help="Layer name for GeoPackage (.gpkg)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100_000,
        help="Max number of features to load (default: 100000)"
    )
    parser.add_argument(
        "--simplify",
        type=str,                 # <— accept "off" or a number as string
        default="0.01",           # keep the same default, but as a string
        help="Simplify tolerance for polygons/lines; number like '0.01' or 'off' to disable"
    )
    parser.add_argument(
    "--point-size",
    type=float,
    default=None,
    help="Point size in pixels (overrides automatic sizing)"
    )

    args = parser.parse_args()

    default_limit = 100_000

    app = QApplication(sys.argv)
    win = VectorViewer(args.path, args.column, args.limit, args.simplify, args.layer,point_size=args.point_size)
    win.show()
    app.processEvents()
    win.raise_()
    win.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)
