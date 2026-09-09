"""Verify surface coordinates, slice spectra and existing-window integration."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QFontDatabase
from src.ui_stft3d import StftSurfaceWindow


class SurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"
        if font.exists():
            QFontDatabase.addApplicationFont(str(font))
        cls.app.setFont(QFont("Segoe UI", 9))

    def setUp(self):
        self.window = StftSurfaceWindow()
        self.times = np.linspace(0.25, 2.25, 25)
        self.freqs = np.linspace(-100, 300, 81)
        self.mag = np.exp(-self.times[None, :] / 0.8) * (
            np.exp(-((self.freqs[:, None] - 100) / 12) ** 2)
            + 0.4 * np.exp(-((self.freqs[:, None] - 200) / 15) ** 2))

    def tearDown(self):
        self.window.timer.stop()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_slice_coordinates_and_shared_amplitudes(self):
        original = self.mag.copy()
        self.window.set_data(self.times, self.freqs, self.mag, (0, 250))
        mask = (self.freqs >= 0) & (self.freqs <= 250)
        self.assertEqual(len(self.window.slice_artists), 12)
        for index, line in zip(self.window.slice_indices, self.window.slice_artists):
            x, y, z = line.get_data_3d()
            np.testing.assert_allclose(x, self.times[index])
            np.testing.assert_array_equal(y, self.freqs[mask])
            np.testing.assert_array_equal(z, self.mag[mask, index])
        self.assertEqual(self.window.axis.get_xlabel(), "Time (s)")
        self.assertEqual(self.window.axis.get_ylabel(), "Frequency (Hz)")
        np.testing.assert_array_equal(self.mag, original)

    def test_log_scale_and_camera_preservation(self):
        self.window.set_data(self.times, self.freqs, self.mag, (0, 250), folded=True)
        self.window.axis.view_init(elev=45, azim=20)
        self.window.log_scale.setChecked(True)
        self.window.redraw()
        _, _, z = self.window.slice_artists[0].get_data_3d()
        mask = (self.freqs >= 0) & (self.freqs <= 250)
        np.testing.assert_allclose(z, 20 * np.log10(self.mag[mask, 0] + 1e-12))
        self.assertEqual(self.window.axis.elev, 45)
        self.assertEqual(self.window.axis.azim, 20)
        self.window.show_lines.setChecked(False)
        self.window.redraw()
        self.assertEqual(self.window.slice_artists, [])
        self.window.reset_view()
        self.assertEqual(self.window.axis.elev, 28)
        self.window.invalidate()
        self.assertIsNone(self.window.surface)

    def test_insufficient_and_invalid_data(self):
        with self.assertRaises(ValueError):
            self.window.set_data(self.times[:1], self.freqs, self.mag[:, :1], (0, 250))
        with self.assertRaises(ValueError):
            self.window.set_data(self.times, self.freqs, self.mag, (1000, 2000))
        with self.assertRaises(ValueError):
            self.window.set_data(self.times, self.freqs, self.mag.T, (0, 250))

    def test_dense_surface_keeps_full_resolution_slice_lines(self):
        frequencies = np.linspace(0, 300, 1000)
        times = np.linspace(0, 1, 4)
        magnitude = np.ones((1000, 4))
        magnitude[501, :] = [5, 4, 3, 2]
        self.window.set_data(times, frequencies, magnitude, (0, 300))
        self.assertIn("sampled for display", self.window.caption.text())
        self.assertEqual(len(self.window.slice_artists), 4)
        for index, line in enumerate(self.window.slice_artists):
            _, y, z = line.get_data_3d()
            self.assertEqual(len(y), 1000)
            self.assertEqual(z[501], magnitude[501, index])

    def test_main_window_stft_integration(self):
        from src.ui_main import MainWindow
        main = MainWindow()
        try:
            fs = 1000.0
            time = np.arange(4000) / fs
            signal = np.exp(-time / 1.1) * np.cos(2 * np.pi * 90 * time)
            signal += 0.65 * np.exp(-time / 0.5) * np.cos(2 * np.pi * 190 * time)
            main.current_stft_data = signal
            main.current_processed_time = signal
            main.loader_sampling_rate = fs
            main.spec_window_size.spinbox.setValue(256)
            main.freq_min.setValue(30)
            main.freq_max.setValue(260)
            with patch("src.ui_main.QMessageBox.warning", side_effect=AssertionError), patch(
                "src.ui_main.QMessageBox.critical", side_effect=AssertionError
            ):
                main.show_stft_surface()
            surface = main.stft_surface_window
            self.assertIsNotNone(surface)
            self.assertTrue(surface.isVisible())
            self.assertGreater(len(surface.slice_artists), 1)
            np.testing.assert_array_equal(main.current_stft_data, signal)
            self.assertIsNot(surface.figure, main.fig_stft)
            self.app.processEvents()
            output = Path(__file__).resolve().parents[1] / ".preview"
            output.mkdir(exist_ok=True)
            surface.canvas.draw()
            surface.grab().save(str(output / "stft3d-surface.png"))
            surface.close()
            main.show_stft_surface()
            self.assertIs(main.stft_surface_window, surface)
        finally:
            main.update_timer.stop()
            if main.stft_surface_window is not None:
                main.stft_surface_window.timer.stop()
                main.stft_surface_window.close()
            main.close()
            main.deleteLater()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
