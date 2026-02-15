"""Unit tests for exact solution generation."""

import unittest

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover
    np = None
    NUMPY_IMPORT_ERROR = exc
else:
    NUMPY_IMPORT_ERROR = None

try:
    import synthetic_problem
except Exception as exc:  # pragma: no cover
    synthetic_problem = None
    MODULE_IMPORT_ERROR = exc
else:
    MODULE_IMPORT_ERROR = None


@unittest.skipIf(
    NUMPY_IMPORT_ERROR is not None,
    f"NumPy is unavailable in the environment: {NUMPY_IMPORT_ERROR}",
)
@unittest.skipIf(
    MODULE_IMPORT_ERROR is not None,
    f"synthetic_problem import failed: {MODULE_IMPORT_ERROR}",
)
class ExactSolutionTests(unittest.TestCase):
    def test_returns_callable_components(self):
        f_lbd, y_d_lbd, y_fun_lbd, p_lbd, u_lbd = (
            synthetic_problem.build_synthetic_problem(
                alpha=1e-3,
                nu=1e-2,
            )
        )

        for fn in [f_lbd, y_d_lbd, y_fun_lbd, p_lbd, u_lbd]:
            self.assertTrue(callable(fn))

        x = np.array([0.25, 1.0, 1.75])
        y = np.array([0.25, 1.0, 1.75])

        f_vals = f_lbd(x, y)
        y_d_vals = y_d_lbd(x, y)
        u_vals = u_lbd(x, y)

        self.assertEqual(f_vals.shape, x.shape)
        self.assertEqual(y_d_vals.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(f_vals)))
        self.assertTrue(np.all(np.isfinite(y_d_vals)))
        self.assertTrue(
            np.all(np.logical_or(np.isclose(u_vals, 0.0), np.isclose(u_vals, 1.0)))
        )


if __name__ == "__main__":
    unittest.main()
