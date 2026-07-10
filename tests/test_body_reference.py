import unittest
import numpy as np

from step2_body_reference import transform_to_body_frame


class BodyReferenceTests(unittest.TestCase):
    def test_transform_to_body_frame(self):
        origin = np.array([0.0, 0.0, 0.0])
        point = np.array([1.0, 0.0, 0.0])
        x_axis = np.array([1.0, 0.0, 0.0])
        y_axis = np.array([0.0, 1.0, 0.0])
        z_axis = np.array([0.0, 0.0, 1.0])

        coords = transform_to_body_frame(origin, point, x_axis, y_axis, z_axis)
        np.testing.assert_allclose(coords, np.array([1.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
