"""Depth-tested surface rasterization, composited once over Matplotlib axes.

This is visualization-only code. It never modifies the STFT analysis arrays.
Projected coordinates use a bottom-left origin; smaller projected z is nearer.
"""

import numpy as np
from matplotlib.artist import Artist
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import proj3d


def front_fragments(vertices, triangles, width, height, batch_size=250000):
    """Return nearest depth and triangle ID at each pixel center.

    Candidate pixels are processed in bounded batches, independent of face
    draw order. Barycentric depth resolves overlapping or crossing triangles.
    """
    faces = np.asarray(vertices, dtype=float)[triangles]
    depth = np.full(width * height, np.inf)
    owner = np.full(width * height, -1, dtype=np.int32)
    lower = np.maximum(np.floor(faces[:, :, :2].min(axis=1)).astype(int), 0)
    upper = np.minimum(np.ceil(faces[:, :, :2].max(axis=1)).astype(int), [width, height])
    sizes = np.maximum(upper - lower, 0)
    counts = sizes[:, 0] * sizes[:, 1]
    cumulative = np.cumsum(counts, dtype=np.int64)
    starts = cumulative - counts
    total = int(cumulative[-1]) if len(cumulative) else 0
    for offset in range(0, total, batch_size):
        candidate = np.arange(offset, min(total, offset + batch_size), dtype=np.int64)
        face_ids = np.searchsorted(cumulative, candidate, side="right")
        relative = candidate - starts[face_ids]
        px = lower[face_ids, 0] + relative % sizes[face_ids, 0]
        py = lower[face_ids, 1] + relative // sizes[face_ids, 0]
        a, b, c = faces[face_ids].transpose(1, 0, 2)
        denominator = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        valid = np.abs(denominator) > 1e-12
        denominator = np.where(valid, denominator, 1.0)
        u = ((b[:, 1] - c[:, 1]) * (px + 0.5 - c[:, 0]) + (c[:, 0] - b[:, 0]) * (py + 0.5 - c[:, 1])) / denominator
        v = ((c[:, 1] - a[:, 1]) * (px + 0.5 - c[:, 0]) + (a[:, 0] - c[:, 0]) * (py + 0.5 - c[:, 1])) / denominator
        valid &= (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        pixel = (py * width + px)[valid]
        z = (u * a[:, 2] + v * b[:, 2] + (1 - u - v) * c[:, 2])[valid]
        ids = face_ids[valid]
        # Resolve all competing fragments before assigning any colors.
        np.minimum.at(depth, pixel, z)
        winners = z == depth[pixel]
        owner[pixel[winners]] = ids[winners]
    return depth.reshape(height, width), owner.reshape(height, width)


def front_layer_image(owner, colors, opacity):
    """Apply opacity exactly once to the winning face, never to hidden faces."""
    rgba = np.zeros((*owner.shape, 4), dtype=np.uint8)
    covered = owner >= 0
    rgba[covered, :3] = np.round(np.clip(np.asarray(colors)[owner[covered], :3], 0, 1) * 255).astype(np.uint8)
    rgba[covered, 3] = round(np.clip(opacity, 0, 1) * 255)
    return rgba


class FrontLayerSurface(Artist):
    """A 3D surface resolved with a depth buffer, then alpha-composited once."""

    def __init__(self, axis, x, y, z, mappable, opacity, slices):
        super().__init__()
        self.axes = axis
        self.vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        rows, cols = z.shape
        cells = (np.arange(rows - 1)[:, None] * cols + np.arange(cols - 1)).ravel()
        self.triangles = np.concatenate((np.column_stack((cells, cells + 1, cells + cols)),
                                         np.column_stack((cells + 1, cells + cols + 1, cells + cols))))
        self.colors = mappable.to_rgba(self.vertices[self.triangles, 2].mean(axis=1))
        self.opacity = opacity
        self.slices = slices
        self._cache_key = None
        self._cache_image = None
        self.set_zorder(2.5)

    def _project(self, points, origin):
        x, y, z = proj3d.proj_transform(*points.T, self.axes.get_proj())
        xy = self.axes.transData.transform(np.column_stack((x, y))) - origin
        return np.column_stack((xy, z))

    def _draw_visible_slices(self, rgba, depth, origin):
        height, width = depth.shape
        line_color = np.round(np.array(to_rgba("#142634")) * 255).astype(np.uint8)
        finite_depth = depth[np.isfinite(depth)]
        tolerance = max(1e-6, np.ptp(finite_depth) / min(width, height)) if finite_depth.size else 1e-6
        for line in self.slices:
            projected = self._project(np.column_stack(line.get_data_3d()), origin)
            for a, b in zip(projected[:-1], projected[1:]):
                if not np.isfinite([a, b]).all():
                    continue
                steps = max(2, int(np.ceil(np.max(np.abs(b[:2] - a[:2])) * 2)) + 1)
                # A segment outside the axes cannot contribute a slice pixel.
                if np.any(np.maximum(a[:2], b[:2]) < 0) or np.any(np.minimum(a[:2], b[:2]) >= [width, height]):
                    continue
                fraction = np.linspace(0, 1, min(steps, 4 * max(width, height)))
                samples = a + fraction[:, None] * (b - a)
                px, py = np.floor(samples[:, :2]).astype(int).T
                valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
                px, py, zs = px[valid], py[valid], samples[valid, 2]
                visible = zs <= depth[py, px] + tolerance
                rgba[py[visible], px[visible]] = line_color

    def draw(self, renderer):
        if not self.get_visible():
            return
        bbox = self.axes.bbox
        origin = np.floor([bbox.x0, bbox.y0])
        width, height = np.ceil([bbox.x1, bbox.y1]).astype(int) - origin.astype(int)
        if width < 1 or height < 1:
            return
        key = (self.axes.get_proj().tobytes(), self.axes.transData.get_affine().get_matrix().tobytes(),
               width, height, tuple(origin))
        if key != self._cache_key:
            projected = self._project(self.vertices, origin)
            depth, owner = front_fragments(projected, self.triangles, width, height)
            rgba = front_layer_image(owner, self.colors, self.opacity)
            self._draw_visible_slices(rgba, depth, origin)
            # RendererBase.draw_image places row zero at the lower edge.
            self._cache_image = np.ascontiguousarray(rgba)
            self._cache_key = key
        gc = renderer.new_gc()
        try:
            gc.set_clip_rectangle(bbox)
            renderer.draw_image(gc, origin[0], origin[1], self._cache_image)
        finally:
            gc.restore()
        self.stale = False
