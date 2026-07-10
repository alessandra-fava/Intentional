import unittest
import numpy as np

from step3_record_reachability import ExponentialMovingAverage


class SmoothingTests(unittest.TestCase):
    def test_ema_blends_new_measurements(self):
        smoother = ExponentialMovingAverage(alpha=0.5)
        first = smoother.update(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        second = smoother.update(np.array([2.0, 0.0, 0.0], dtype=np.float32))

        np.testing.assert_allclose(first, np.array([0.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(second, np.array([1.0, 0.0, 0.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
