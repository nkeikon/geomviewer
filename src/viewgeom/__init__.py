#!/usr/bin/env python3
"""
viewgeom — Interactive viewer for vector datasets (.shp, .geojson, .gpkg, .parquet, .geoparquet)
"""

import sys, os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import GeometryCollection
from shapely.ops import unary_union
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QGraphicsPathItem, QGraphicsEllipseItem, QStatusBar, QScrollBar
)
from PySide6.QtGui import QPen, QColor, QPainterPath, QPainter
from PySide6.QtCore import Qt

__version__ = "0.1.1"


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
            import pyarrow  # noqa
        except ImportError:
            raise ImportError("Reading Parquet requires 'pyarrow'. Install via: pip install pyarrow")
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
                print(f"[INFO] Loading GPKG layer: {layer}")
                gdf = gpd.read_file(path, layer=layer)

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

    else:
        raise ValueError(f"Unsupported format: {ext}")

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

    # --- Limit features for very large sets ---
    n = len(gdf)
    if n > limit:
        print(f"[WARN] Large dataset ({n:,} features) — sampling {limit:,}")
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
# Color mapping (numeric only)
# ---------------------------------------------------------------------
def get_color_mapping(gdf, column, cmap_name="viridis"):
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
    else:
        colors = ["yellow"] * len(gdf)
    return colors, None


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
    def __init__(self, path, column=None, limit=100_000, simplify=0.01, layer=None):
        super().__init__()
        # print(f"[INFO] Loading {path}")
        self.path = path
        self.layer = layer  

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
        extent = max(maxx - minx, maxy - miny)
        self.point_size = max(1.5, min(6, extent * 0.001))

        self.colormaps = [
            "plasma",    # default continuous
            "turbo",     # bold, web-mapping look
            "cividis",   # accessible & balanced
            "Spectral",  # diverging / strong variation
            "tab10"      # categorical mode fallback
        ]

        self.cmap_index = 0

        self.scene = QGraphicsScene(self)
        self.view = VectorView(self.scene)
        self.setCentralWidget(self.view)
        self.setStatusBar(QStatusBar())
        self.basemap_items = []
        self.feature_items = []

        # ---- Load global basemap ----
        self._load_basemap()

        # ---- Numeric column selection ----
        num_cols = [c for c in self.gdf.columns if self.gdf[c].dtype.kind in "if"]
        if not num_cols:
            # print("[WARN] No numeric columns found — using uniform color.")
            self.color_col = None
        else:
            if column and column in num_cols:
                self.color_col = column
            else:
                print("[INFO] Numeric columns detected:")
                for i, c in enumerate(num_cols):
                    print(f"  [{i}] {c}")
                choice = input("Select column index: ").strip()
                try:
                    idx = int(choice)
                    self.color_col = num_cols[idx]
                except Exception:
                    print("[INFO] Invalid selection, using first column.")
                    self.color_col = num_cols[0]
                if self.color_col:
                    print(f"[INFO] Coloring by: {self.color_col}")

        self._update_window_title()

        # ---- Color mapping ----
        if self.color_col:
            self.gdf["_color"], _ = get_color_mapping(self.gdf, self.color_col)
        else:
            self.gdf["_color"] = ["yellow"] * len(self.gdf)

        # ---- Basemap logic ----
        if self.color_col:
            # Automatically load basemap if coloring by numeric data
            self._load_basemap()
            self._draw_basemap()
        else:
            # Skip basemap by default when only boundaries are shown
            self.base_gdf = None
            print("[INFO] Basemap optional (Press 'B' to add)")

        # ---- Draw features ----
        self._draw_geoms()

        # ---- Scene extents ----
        minx, miny, maxx, maxy = self.gdf.total_bounds
        self.scene.setSceneRect(minx, -maxy, maxx - minx, maxy - miny)
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.resize(1000, 800)
        print(f"[INFO] Features displayed: {len(self.gdf):,}")

    # -----------------------------------------------------------------
    def _load_basemap(self):
        try:
            self.base_gdf = gpd.read_file(
                "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
            )
            if self.base_gdf.crs != self.gdf.crs:
                self.base_gdf = self.base_gdf.to_crs(self.gdf.crs)
            print("[INFO] Basemap loaded")
        except Exception as e:
            print(f"[WARN] Could not load basemap: {e}")
            self.base_gdf = None

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
            # item.setZValue(10)
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

        for _, row in self.gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            # Choose color and brush per feature
            if self.color_col:
                color = self._color_for_value(row["_color"])
                brush = color
            else:
                # No numeric column: outline only, use strong contrasting color
                palette = QApplication.palette()
                bg = palette.window().color()
                brightness = (bg.red() * 299 + bg.green() * 587 + bg.blue() * 114) / 1000

                # Choose outline color depending on theme
                if brightness < 128:
                    # Dark background → bright red
                    color = QColor(255, 80, 80)   # vivid red
                else:
                    # Light background → dark red or blue for better contrast
                    color = QColor(150, 0, 0)     # deep red
                    # or try: color = QColor(0, 70, 200)  # navy blue alternative

                brush = Qt.BrushStyle.NoBrush

            pen.setColor(color)

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
        num_cols = [c for c in self.gdf.columns if self.gdf[c].dtype.kind in "if"]
        if not num_cols:
            print("[INFO] No numeric columns available to switch.")
            return

        # current index
        idx = num_cols.index(self.color_col) if self.color_col in num_cols else 0

        # wrap around cycling
        idx = (idx + direction) % len(num_cols)
        self.color_col = num_cols[idx]
        print(f"[INFO] Coloring by: {self.color_col}")
        self._update_window_title()

        # Keep the currently selected colormap
        cmap_name = self.colormaps[self.cmap_index]
        self.gdf["_color"], _ = get_color_mapping(self.gdf, self.color_col, cmap_name=cmap_name)

        # redraw
        self._draw_geoms()
        if hasattr(self, "basemap_items") and self.basemap_items:
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
            self.cmap_index = (self.cmap_index + 1) % len(self.colormaps)
            cmap_name = self.colormaps[self.cmap_index]
            print(f"[INFO] Switched colormap to: {cmap_name}")
            if self.color_col:
                self.gdf["_color"], _ = get_color_mapping(self.gdf, self.color_col, cmap_name=cmap_name)
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
                # Basemap not drawn → load and draw
                if self.base_gdf is None:
                    self._load_basemap()
                self._draw_basemap()
                # print("[INFO] Basemap added")
        elif k == Qt.Key.Key_R:
            self.view.resetTransform()
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
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

    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = VectorViewer(args.path, args.column, args.limit, args.simplify, args.layer)
    win.show()
    app.processEvents()
    win.raise_()
    win.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
