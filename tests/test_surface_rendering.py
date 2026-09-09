"""Numerical pixel checks for nearest-layer transparency."""

import unittest
from io import BytesIO
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from src.surface_rendering import front_fragments, front_layer_image, FrontLayerSurface


class FrontLayerTests(unittest.TestCase):
    def test_hidden_faces_never_accumulate_alpha_or_color(self):
        vertices = np.array([[1, 1, -2], [9, 1, -2], [1, 9, -2],
                             [1, 1, -1], [9, 1, -1], [1, 9, -1]], dtype=float)
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        colors = np.array([[1, 0, 0, 1], [0, 0, 1, 1]])
        for order in ([0, 1], [1, 0]):
            _, owner = front_fragments(vertices, faces[order], 12, 12, batch_size=7)
            image = front_layer_image(owner, colors[order], .5)
            np.testing.assert_array_equal(image[3, 3], [255, 0, 0, 128])
            np.testing.assert_array_equal(image[11, 11], [0, 0, 0, 0])
            # Composite once with a background/grid pixel, not the hidden blue face.
            alpha = image[3, 3, 3] / 255
            composite = alpha * image[3, 3, :3] + (1 - alpha) * np.array([80, 80, 80])
            self.assertLess(composite[2], 41)

    def test_intersections_use_pixel_depth_not_average_face_order(self):
        vertices = np.array([[1, 1, -3], [10, 1, -1], [1, 10, -3],
                             [1, 1, -2], [10, 1, -2], [1, 10, -2]], dtype=float)
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        depth, owner = front_fragments(vertices, faces, 12, 12)
        self.assertEqual(owner[2, 2], 0)
        self.assertEqual(owner[2, 7], 1)
        self.assertAlmostEqual(depth[2, 7], -2)

    def test_degenerate_and_outside_faces_are_empty(self):
        vertices = np.array([[1, 1, 0], [2, 2, 0], [3, 3, 0],
                             [20, 20, 0], [21, 20, 0], [20, 21, 0]])
        _, owner = front_fragments(vertices, np.array([[0, 1, 2], [3, 4, 5]]), 10, 10)
        self.assertTrue(np.all(owner == -1))

    def test_actual_canvas_orientation_rotation_resize_and_export(self):
        figure = Figure(figsize=(4, 4), dpi=100)
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_subplot(111, projection="3d")
        axis.set_axis_off()
        x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 12))
        z = np.exp(-((x - .3) ** 2 + (y - .4) ** 2) * 20)
        axis.auto_scale_xyz(x, y, z)
        artist = FrontLayerSurface(axis, x, y, z, ScalarMappable(norm=Normalize(0, 1), cmap="viridis"), .65, [])
        axis.add_artist(artist)
        previous_key = None
        for elev, azim in [(28, -58), (35, 120), (-20, 40)]:
            axis.view_init(elev=elev, azim=azim)
            canvas.draw()
            self.assertNotEqual(artist._cache_key, previous_key)
            previous_key = artist._cache_key
            image = artist._cache_image
            samples = np.argwhere(image[:, :, 3] > 0)
            self.assertGreater(len(samples), 100)
            py, px = samples[len(samples) // 3]
            origin = np.floor([axis.bbox.x0, axis.bbox.y0]).astype(int)
            actual = np.asarray(canvas.buffer_rgba())
            pixel = actual[actual.shape[0] - 1 - (origin[1] + py), origin[0] + px, :3]
            alpha = image[py, px, 3] / 255
            expected = alpha * image[py, px, :3] + (1 - alpha) * 255
            np.testing.assert_allclose(pixel, expected, atol=2)
        old_size = artist._cache_image.shape
        figure.set_size_inches(5, 4)
        canvas.draw()
        self.assertNotEqual(artist._cache_key, previous_key)
        for format_name in ("png", "svg"):
            output = BytesIO()
            figure.savefig(output, format=format_name, dpi=150)
            self.assertGreater(len(output.getvalue()), 1000)

    def test_slice_lines_behind_surface_are_hidden(self):
        from types import SimpleNamespace
        x, y = np.meshgrid([0, 1], [0, 1])
        figure = Figure()
        axis = figure.add_subplot(111, projection="3d")
        for line_z, should_draw in [(-1, False), (-3, True)]:
            line = SimpleNamespace(get_data_3d=lambda: ([2, 7], [3, 3], [line_z, line_z]))
            artist = FrontLayerSurface(axis, x, y, x * 0, ScalarMappable(), .65, [line])
            artist._project = lambda points, origin: points
            rgba = np.zeros((10, 10, 4), dtype=np.uint8)
            artist._draw_visible_slices(rgba, np.full((10, 10), -2.0), (0, 0))
            self.assertEqual(bool(rgba[:, :, 3].any()), should_draw)


if __name__ == "__main__":
    unittest.main()
