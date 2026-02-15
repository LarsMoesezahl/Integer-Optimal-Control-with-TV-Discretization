"""Unit tests for mesh helper utilities in rt0_dual_solver."""

import unittest

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover
    np = None
    NUMPY_IMPORT_ERROR = exc
else:
    NUMPY_IMPORT_ERROR = None

try:
    from rt0_dual_solver import get_cell_coordinates, mid_points
except Exception as exc:  # pragma: no cover
    get_cell_coordinates = None
    mid_points = None
    MODULE_IMPORT_ERROR = exc
else:
    MODULE_IMPORT_ERROR = None


class MeshHelperTests(unittest.TestCase):
    def setUp(self):
        if NUMPY_IMPORT_ERROR is not None:
            self.skipTest(
                f"NumPy is unavailable in the environment: {NUMPY_IMPORT_ERROR}"
            )
        if MODULE_IMPORT_ERROR is not None:
            self.skipTest(f"rt0_dual_solver import failed: {MODULE_IMPORT_ERROR}")

    @staticmethod
    def _fake_mesh():
        class _FakeConnectivity:
            @staticmethod
            def links(_cell):
                return np.array([0, 1, 2, 3], dtype=np.int32)

        class _FakeTopology:
            dim = 2

            @staticmethod
            def connectivity(_from_dim, _to_dim):
                return _FakeConnectivity()

        class _FakeGeometry:
            x = np.array(
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                ]
            )

        class _FakeMesh:
            topology = _FakeTopology()
            geometry = _FakeGeometry()

        return _FakeMesh()

    def test_get_cell_coordinates_returns_four_vertices(self):
        q1, q2, q3, q4 = get_cell_coordinates(self._fake_mesh(), 0)
        np.testing.assert_allclose(q1, [0.0, 0.0])
        np.testing.assert_allclose(q4, [1.0, 1.0])

    def test_mid_points_order_and_values(self):
        mids = mid_points(1, self._fake_mesh())
        self.assertEqual(len(mids), 1)
        left, bottom, top, right = mids[0]
        np.testing.assert_allclose(left, [0.0, 0.5])
        np.testing.assert_allclose(bottom, [0.5, 0.0])
        np.testing.assert_allclose(top, [0.5, 1.0])
        np.testing.assert_allclose(right, [1.0, 0.5])


if __name__ == "__main__":
    unittest.main()
